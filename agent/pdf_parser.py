"""PDF 文档预处理器

将原始 PDF 解析为结构化文本，保留标题、章节、表格、条款编号等信息。
使用 pymupdf + pdfplumber 双引擎解析，提升表格和版面还原质量。
"""
import os
import json
import fitz  # pymupdf
import pdfplumber
from agent.config import RAW_DIR, PROCESSED_DIR


def parse_pdf_with_pymupdf(pdf_path: str) -> dict:
    """使用 PyMuPDF 解析 PDF，提取文本和结构信息"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        # 提取表格（PyMuPDF 基础表格提取）
        tables = []
        for tab in page.find_tables():
            tables.append(tab.extract())
        pages.append({
            "page_num": page_num,
            "text": text,
            "tables": tables,
        })
    doc.close()
    return {
        "source": pdf_path,
        "parser": "pymupdf",
        "pages": pages,
    }


def parse_pdf_with_pdfplumber(pdf_path: str) -> dict:
    """使用 pdfplumber 解析 PDF，更擅长表格提取"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            tables = []
            for table in page.extract_tables():
                # 将表格转为文本行
                table_lines = []
                for row in table:
                    row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                    table_lines.append(row_text)
                tables.append(table_lines)
            pages.append({
                "page_num": page_num,
                "text": text,
                "tables": tables,
            })
    return {
        "source": pdf_path,
        "parser": "pdfplumber",
        "pages": pages,
    }


def merge_parse_results(pymupdf_result: dict, plumber_result: dict) -> dict:
    """合并两个解析器的结果，取各自优势"""
    merged_pages = []
    py_pages = {p["page_num"]: p for p in pymupdf_result["pages"]}
    pl_pages = {p["page_num"]: p for p in plumber_result["pages"]}

    all_page_nums = sorted(set(py_pages.keys()) | set(pl_pages.keys()))

    for pn in all_page_nums:
        py_page = py_pages.get(pn, {})
        pl_page = pl_pages.get(pn, {})

        # 文本：优先用 pdfplumber（表格区域文本更好），补充 pymupdf
        text = pl_page.get("text", "") or py_page.get("text", "")
        
        # 表格：优先用 pdfplumber（表格提取更强）
        tables = pl_page.get("tables", [])
        if not tables:
            tables = py_page.get("tables", [])

        merged_pages.append({
            "page_num": pn,
            "text": text,
            "tables": tables,
        })

    return {
        "source": pymupdf_result["source"],
        "parser": "merged",
        "pages": merged_pages,
    }


def process_single_pdf(pdf_path: str, output_path: str) -> dict:
    """处理单个 PDF 文件，输出结构化 JSON"""
    print(f"  解析: {os.path.basename(pdf_path)}")
    
    # 双引擎解析
    pymupdf_result = parse_pdf_with_pymupdf(pdf_path)
    plumber_result = parse_pdf_with_pdfplumber(pdf_path)
    
    # 合并结果
    merged = merge_parse_results(pymupdf_result, plumber_result)
    
    # 生成全文
    full_text_parts = []
    for page in merged["pages"]:
        full_text_parts.append(f"--- 第 {page['page_num']} 页 ---\n")
        full_text_parts.append(page["text"])
        if page["tables"]:
            for i, table in enumerate(page["tables"], 1):
                full_text_parts.append(f"\n[表格 {i}]")
                if isinstance(table, list):
                    for row in table:
                        full_text_parts.append(row if isinstance(row, str) else str(row))
                else:
                    full_text_parts.append(str(table))
    
    merged["full_text"] = "\n".join(full_text_parts)
    merged["total_pages"] = len(merged["pages"])
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    return merged


def get_doc_id_from_filename(filename: str) -> str:
    """从文件名推导 doc_id"""
    name = os.path.splitext(filename)[0]
    return name


def preprocess_domain(domain: str):
    """预处理某个领域的所有 PDF"""
    domain_raw_dir = os.path.join(RAW_DIR, domain)
    domain_out_dir = os.path.join(PROCESSED_DIR, domain)
    os.makedirs(domain_out_dir, exist_ok=True)

    if not os.path.exists(domain_raw_dir):
        print(f"  目录不存在，跳过: {domain_raw_dir}")
        return

    # 处理 PDF 文件
    pdf_files = [f for f in os.listdir(domain_raw_dir) 
                 if f.lower().endswith(('.pdf', '.PDF'))]
    
    # 处理子目录中的文件（如 regulatory/attachments, regulatory/html, regulatory/txt）
    for root, dirs, files in os.walk(domain_raw_dir):
        for f in files:
            if f.lower().endswith(('.pdf', '.PDF')) and f not in pdf_files:
                pdf_files.append(os.path.relpath(os.path.join(root, f), domain_raw_dir))

    print(f"\n[{domain}] 发现 {len(pdf_files)} 个 PDF 文件")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(domain_raw_dir, pdf_file)
        doc_id = get_doc_id_from_filename(pdf_file)
        output_path = os.path.join(domain_out_dir, f"{doc_id}.json")
        
        if os.path.exists(output_path):
            print(f"  已存在，跳过: {doc_id}")
            continue
        
        try:
            process_single_pdf(pdf_path, output_path)
        except Exception as e:
            print(f"  解析失败: {pdf_file}, 错误: {e}")


def preprocess_regulatory_special():
    """特殊处理监管法规领域的 html/txt 文件"""
    html_dir = os.path.join(RAW_DIR, "regulatory", "html")
    txt_dir = os.path.join(RAW_DIR, "regulatory", "txt")
    out_dir = os.path.join(PROCESSED_DIR, "regulatory")
    os.makedirs(out_dir, exist_ok=True)

    # 处理 HTML 文件
    if os.path.exists(html_dir):
        for f in os.listdir(html_dir):
            if f.endswith(('.html', '.htm')):
                html_path = os.path.join(html_dir, f)
                doc_id = get_doc_id_from_filename(f)
                output_path = os.path.join(out_dir, f"{doc_id}.json")
                
                if os.path.exists(output_path):
                    continue
                
                try:
                    from bs4 import BeautifulSoup
                    with open(html_path, "r", encoding="utf-8") as fh:
                        soup = BeautifulSoup(fh.read(), "html.parser")
                        text = soup.get_text(separator="\n", strip=True)
                    result = {
                        "source": html_path,
                        "parser": "beautifulsoup",
                        "full_text": text,
                        "total_pages": 1,
                        "pages": [{"page_num": 1, "text": text, "tables": []}],
                    }
                    with open(output_path, "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    print(f"  解析 HTML: {doc_id}")
                except Exception as e:
                    print(f"  HTML 解析失败: {f}, 错误: {e}")

    # 处理 TXT 文件
    if os.path.exists(txt_dir):
        for f in os.listdir(txt_dir):
            if f.endswith('.txt'):
                txt_path = os.path.join(txt_dir, f)
                doc_id = get_doc_id_from_filename(f)
                output_path = os.path.join(out_dir, f"{doc_id}.json")
                
                if os.path.exists(output_path):
                    continue
                
                try:
                    with open(txt_path, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    result = {
                        "source": txt_path,
                        "parser": "raw_text",
                        "full_text": text,
                        "total_pages": 1,
                        "pages": [{"page_num": 1, "text": text, "tables": []}],
                    }
                    with open(output_path, "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    print(f"  解析 TXT: {doc_id}")
                except Exception as e:
                    print(f"  TXT 解析失败: {f}, 错误: {e}")


def preprocess_all():
    """预处理所有领域的文档"""
    print("=" * 60)
    print("开始文档预处理")
    print("=" * 60)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 处理 PDF 类领域
    for domain in ["insurance", "financial_contracts", "financial_reports", "research"]:
        preprocess_domain(domain)
    
    # 监管法规特殊处理
    preprocess_domain("regulatory")
    preprocess_regulatory_special()
    
    # 统计
    total_docs = 0
    for domain in os.listdir(PROCESSED_DIR):
        domain_dir = os.path.join(PROCESSED_DIR, domain)
        if os.path.isdir(domain_dir):
            total_docs += len([f for f in os.listdir(domain_dir) if f.endswith('.json')])
    
    print(f"\n预处理完成，共生成 {total_docs} 个结构化文档")


if __name__ == "__main__":
    preprocess_all()
