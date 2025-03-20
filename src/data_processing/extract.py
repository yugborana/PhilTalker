from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder, dpi=300):
    """Converts a PDF to images and saves them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, dpi=dpi)
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        print(f"Saved: {img_path}")

if __name__ == "__main__":
    pdf_path = "data/hehe.pdf"  # Change to your actual PDF path
    output_folder = "data/images"
    pdf_to_images(pdf_path, output_folder)
