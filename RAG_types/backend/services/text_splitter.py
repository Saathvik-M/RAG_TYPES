from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:

    def split(self, documents):

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(documents)

        return chunks