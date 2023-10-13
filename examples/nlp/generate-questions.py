from typing import Optional

import crossfit as cf
from crossfit.backend.torch import CurratedTokenizer, CuratedGenerator


class QueryPrompt(cf.ColumnOp):
    def __init__(
        self,
        input_col: str,
        output_col: Optional[str] = None,
        keep_cols=None,
        doc_name: str = "Document",
        query_name: str = "Relevant query",
    ):
        super().__init__(
            input_col=input_col, dtype="str", output_col=output_col, keep_cols=keep_cols
        )
        self.doc_name = doc_name
        self.query_name = query_name

    def call(self, data):
        return f"{self.doc_name}: \n\n" + data + f"\n\n{self.query_name}: \n\n"


def main(model="tiiuae/falcon-7b-instruct", dataset="beir/quora", overwrite=True):
    # model = "meta-llama/Llama-2-7b-hf"
    dataset: cf.IRDataset = cf.load_dataset(dataset, overwrite=False)

    pipe = cf.Sequential(
        QueryPrompt("text"),
        CurratedTokenizer.from_hf_hub(name=model, cols=["text"]),
        # cf.Repartition(50_000),
        CuratedGenerator(model, batch_size=16, batch_steps=10, output_col="answer"),
        keep_cols=["index", "_id", "text"],
    )

    passages = dataset.item.ddf()
    passages = passages.loc[:30]
    generated = pipe(passages).compute()

    for _, row in generated.to_pandas().iterrows():
        print(row["text"])
        print()
        print("Response: " + row["answer"])
        print()
        print()


if __name__ == "__main__":
    main()
