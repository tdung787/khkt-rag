import os
import zipfile
import xml.etree.ElementTree as ET
from tqdm import tqdm

def extract_images_in_order(docx_path, output_folder="docx_images"):
    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(docx_path) as docx_zip:
        # Load relationships
        rels_xml = docx_zip.read("word/_rels/document.xml.rels")
        rels_root = ET.fromstring(rels_xml)
        rels = {}
        for rel in rels_root:
            rid = rel.attrib.get('Id')
            target = rel.attrib.get('Target')
            if target.startswith("media/"):
                rels[rid] = f"word/{target}"

        # Load document.xml
        doc_xml = docx_zip.read("word/document.xml")
        doc_root = ET.fromstring(doc_xml)

        ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
              'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
              'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

        # Tìm tất cả ảnh theo thứ tự xuất hiện
        blips = doc_root.findall('.//a:blip', ns)
        print(f"Found {len(blips)} images in order.")

        for idx, blip in enumerate(tqdm(blips, desc="Images"), start=1):
            rid = blip.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            img_path_in_docx = rels.get(rid)
            if img_path_in_docx:
                ext = os.path.splitext(img_path_in_docx)[1]
                out_path = os.path.join(output_folder, f"image_{idx:02d}{ext}")
                with docx_zip.open(img_path_in_docx) as source, open(out_path, 'wb') as target:
                    target.write(source.read())
                    
    return output_folder

if __name__ == "__main__":
    extract_images_in_order("data/input/docx/ester-lipid_hoa12.docx")
