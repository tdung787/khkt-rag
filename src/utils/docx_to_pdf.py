import subprocess
import os

def docx_to_pdf(input_path, output_dir=None):
    """
    Chuyển đổi file .docx sang .pdf bằng LibreOffice (headless mode).
    Args:
        input_path (str): Đường dẫn đến file DOCX hoặc thư mục.
        output_dir (str, optional): Thư mục đầu ra chứa file PDF.
    """
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file hoặc thư mục: {input_path}")
        return

    # Nếu đầu ra chưa tồn tại, tạo mới
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(os.path.abspath(input_path))

    try:
        # Chạy lệnh LibreOffice để convert
        subprocess.run(
            [
                "libreoffice",
                "--headless",        # chạy nền
                "--convert-to", "pdf",
                "--outdir", output_dir,
                input_path
            ],
            check=True
        )
        print(f"✅ Đã chuyển đổi: {input_path} → {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Lỗi khi chuyển đổi: {e}")


if __name__ == "__main__":
    # Ví dụ: đổi "input.docx" thành file thật của bạn
    input_file = "sinh12.docx"
    output_folder = "./pdf_out"

    docx_to_pdf(input_file, output_folder)
