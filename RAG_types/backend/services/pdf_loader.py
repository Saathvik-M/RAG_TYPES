from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:

    def load_pdf(self, path: str):

        loader = PyPDFLoader(path)

        documents = loader.load()

        return documents