import asyncio
import logging
import os
import time
import urllib.request
from typing import Optional

import aiohttp
import cudf
import dask_cudf
import dask.dataframe as dd
import torchx.runner as runner
import torchx.specs as specs
from dask.distributed import Client
from torchx.components.utils import python as utils_python
from tqdm.asyncio import tqdm_asyncio

from crossfit.dataset.home import CF_HOME


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TGI_IMAGE_NAME = "ghcr.io/huggingface/text-generation-inference"
TGI_IMAGE_VERSION = "1.1.1"


class HFGenerator:
    def __init__(
        self,
        path_or_name,
        image_name: str = TGI_IMAGE_NAME,
        image_version: str = TGI_IMAGE_VERSION,
        num_gpus: int = 1,
        mem_gb: int = 8,
        max_wait_seconds: int = 1800,
        max_tokens: int = 384,
    ) -> None:
        self.path_or_name = path_or_name
        self.image_name = image_name
        self.image_version = image_version
        self.num_gpus = num_gpus
        self.mem_gb = mem_gb
        self.max_wait_seconds = max_wait_seconds
        self.max_tokens = max_tokens

    def __enter__(self):
        self.inference_server = get_tgi_app_def(
            self.path_or_name,
            image_name=self.image_name,
            image_version=self.image_version,
            num_gpus=self.num_gpus,
            mem_gb=self.mem_gb,
        )

        self.runner = runner.get_runner()

        self.app_handle = self.runner.run(
            self.inference_server,
            scheduler="local_docker",
        )

        self.status = self.runner.status(self.app_handle)

        self.container_name = self.app_handle.split("/")[-1]
        self.local_docker_client = self.runner._scheduler_instances["local_docker"]._docker_client
        self.networked_containers = self.local_docker_client.networks.get("torchx").attrs[
            "Containers"
        ]

        self.ip_address = None
        for _, container_config in self.networked_containers.items():
            if self.container_name in container_config["Name"]:
                self.ip_address = container_config["IPv4Address"].split("/")[0]
                break
        if not self.ip_address:
            raise RuntimeError("Unable to get server IP address.")

        self.health = None
        for i in range(self.max_wait_seconds):
            try:
                urllib.request.urlopen(f"http://{self.ip_address}/health")
                self.health = "OK"
            except urllib.error.URLError:
                time.sleep(1)

            if self.health == "OK" or self.status.state != specs.AppState.RUNNING:
                break

            if i % 10 == 1:
                logger.info("Waiting for server to be ready...")

            self.status = self.runner.status(self.app_handle)

        logger.info(self.status)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def stop(self):
        if self.status.state == specs.AppState.RUNNING:
            self.runner.stop(self.app_handle)

    async def infer_async(self, data):
        address = self.ip_address
        async with Client(asynchronous=True) as dask_client:
            tasks = [
                dask_client.submit(fetch_async, address, string, max_tokens=self.max_tokens)
                for string in data
            ]
            return await tqdm_asyncio.gather(*tasks)

    def infer(self, data, col: Optional[str] = None):
        if isinstance(data, dd.DataFrame):
            if not col:
                raise ValueError("Column name must be provided for a dataframe.")
            data = data.compute()[col].to_pandas().tolist()
        generated_text = asyncio.run(self.infer_async(data))

        input_col = col or "inputs"
        output_col = "generated_text"
        npartitions = getattr(data, "npartitions", self.num_gpus)
        ddf = dask_cudf.from_cudf(
            cudf.DataFrame(
                {input_col: data, output_col: generated_text},
                npartitions=npartitions,
            )
        )
        return ddf


async def fetch_async(http_address: str, prompt: int, max_tokens: int):
    async with aiohttp.ClientSession() as session:
        url = f"http://{http_address}/generate"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens},
        }
        async with session.post(url, json=payload) as response:
            response_json = await response.json()
            return response_json["generated_text"]


def get_tgi_app_def(
    path_or_name: str,
    image_name: str = TGI_IMAGE_NAME,
    image_version: str = TGI_IMAGE_VERSION,
    num_gpus: int = 1,
    mem_gb: int = 8,
) -> specs.AppDef:
    if os.path.isabs(path_or_name):
        args = ["/data"]
        mounts = [
            specs.BindMount(path_or_name, "/data"),
        ]
    else:
        args = [path_or_name]
        mounts = []

    if num_gpus > 1:
        args.extend(["--sharded", "true", "--num-shard", f"{num_gpus}"])

    app_def = specs.AppDef(
        name="generator",
        roles=[
            specs.Role(
                name="tgi",
                image=f"{image_name}:{image_version}",
                entrypoint="--model-id",
                args=args,
                num_replicas=1,
                resource=specs.Resource(cpu=1, gpu=num_gpus, memMB=mem_gb * 1024),
                mounts=mounts,
            ),
        ],
    )
    return app_def
