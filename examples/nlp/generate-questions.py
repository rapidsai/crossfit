import crossfit as cf
from crossfit.backend.torch import CurratedTokenizer, CuratedGenerator


class QueryPrompt(cf.ColumnOp[str]):
    doc_name: str = "Document"
    query_name: str = "Relevant query"

    def call(self, data):
        return f"{self.doc_name}: \n\n" + data + f"\n\n{self.query_name}: \n\n"


def main(model="meta-llama/Llama-2-7b-hf", dataset="beir/quora", overwrite=True):
    dataset: cf.IRDataset = cf.load_dataset(dataset, overwrite=False)

    pipe = cf.Sequential(
        QueryPrompt(cols="text"),
        CurratedTokenizer.from_hf_hub(name=model, cols=["text"]),
        # cf.Repartition(50_000),
        CuratedGenerator(model, batch_size=32, batch_steps=10, output_col="answer"),
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
