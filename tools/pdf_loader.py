"""
PDF Loader Module - loads and chunks specification documents
"""
import os
from pathlib import Path
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFLoader:
    def __init__(self, pdf_dir: str = "data/pdfs", chunk_size: int = 512, chunk_overlap: int = 50):
        self.pdf_dir = Path(pdf_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdfs(self) -> List[Dict[str, str]]:
        """Load all PDFs from the pdf_dir and return chunked documents"""
        documents = []

        if not self.pdf_dir.exists():
            print(f"Warning: PDF directory {self.pdf_dir} not found. Using fallback documents.")
            return self._get_fallback_docs()

        for pdf_path in self.pdf_dir.glob("*.pdf"):
            print(f"Loading {pdf_path.name}...")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page_num, page in enumerate(pdf.pages):
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text() or ""

                    chunks = self.splitter.split_text(text)
                    for chunk in chunks:
                        documents.append({
                            "text": chunk,
                            "source": pdf_path.stem,
                            "page": len(documents)
                        })
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        if not documents:
            return self._get_fallback_docs()

        return documents

    @staticmethod
    def _get_fallback_docs() -> List[Dict[str, str]]:
        """Return fallback in-memory documents when no PDFs are available"""
        return [
            {
                "text": "NVMe 2.0 Specification. NVMe supports up to 65535 I/O queues with 65535 commands per queue, enabling massive flash parallelism. The Command Submission Queue (CSQ) and Completion Queue (CQ) provide independent paths for command submission and completion notification.",
                "source": "NVMe_2.0_spec",
                "page": 0
            },
            {
                "text": "SSD Architecture Guide. The FTL (Flash Translation Layer) maps logical block addresses to physical NAND, handles wear levelling, garbage collection, and bad block management. It ensures data reliability and extends the lifetime of NAND flash memory.",
                "source": "SSD_Architecture_Guide",
                "page": 1
            },
            {
                "text": "PCIe 4.0 Specification. PCIe 4.0 gives approximately 2 GB/s per lane. NVMe SSDs use x4 configuration for 8 GB/s total bandwidth. This enables high-speed data transfer between the SSD controller and the host system.",
                "source": "PCIe_4.0_spec",
                "page": 2
            },
            {
                "text": "JEDEC UFS 3.1 Standard. UFS 3.1 supports full-duplex communication at 23.2 Gbps per lane over M-PHY physical layer. It provides superior performance compared to eMMC with a dedicated command and data flow.",
                "source": "JEDEC_UFS_3.1",
                "page": 3
            },
            {
                "text": "NAND Flash Guide. ECC (Error-Correcting Code) detects and corrects bit errors in NAND flash. Uncorrectable ECC errors (UNCEs) indicate block failure and require remapping to spare blocks. The ECC algorithm strength determines the number of correctable errors.",
                "source": "NAND_Flash_Guide",
                "page": 4
            }
        ]
