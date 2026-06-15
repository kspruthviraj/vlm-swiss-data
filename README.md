---
language:
  - de
  - en
  - fr
  - it
task_categories:
  - visual-question-answering
  - image-captioning
  - document-question-answering
tags:
  - switzerland
  - multimodal
  - vlm
  - document-understanding
  - transport
  - news
  - government
  - retail
size_categories:
  - n<1K
---

# Swiss Multimodal Document Dataset for VLM Fine-Tuning

## Dataset Description

A multimodal dataset of Swiss public documents, transport data, news articles, and product catalogs designed for Vision-Language Model (VLM) fine-tuning. The dataset pairs images with text content and question-answer pairs across multiple Swiss data domains.

### Supported Tasks

- **Visual Question Answering (VQA)**: Answer questions about Swiss documents, timetables, and product images
- **Document Understanding**: Parse and understand Swiss government forms, transport maps, and news articles
- **Image Captioning**: Generate descriptions of Swiss stations, products, and news imagery
- **Multilingual Understanding**: Process content in German, French, Italian, and English

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `image` | Image | Associated image (station photo, PDF page render, product image, news photo) |
| `text` | string | Extracted text content from the source document |
| `language` | string | Language code (`de`, `en`, `fr`, `it`) |
| `source` | string | Data source identifier |
| `category` | string | Content category |
| `qa_pairs` | Sequence | List of `{question, answer}` pairs for VQA training |
| `metadata` | string | JSON-encoded metadata (URLs, IDs, etc.) |

### Data Splits

| Split | Proportion | Description |
|-------|------------|-------------|
| train | 80% | Training examples |
| val | 10% | Validation examples |
| test | 10% | Test examples |

Target: 500–1000 total examples.

## Data Sources

### 1. SBB (Swiss Federal Railways)
- **Category**: `transport`, `timetable`
- **Content**: Station information (IDs, coordinates), train timetables, connection details
- **Images**: Station photos from Wikimedia Commons
- **API**: `transport.opendata.ch` (public, no auth required)

### 2. ZVV (Zürich Transport Network)
- **Category**: `zone_map`, `network_plan`, `stop_info`
- **Content**: Zone maps (PDF), network plans (PDF), stop line information
- **Images**: First-page renders of ZVV PDF maps
- **API**: `transport.opendata.ch` for stop data

### 3. admin.ch (Swiss Federal Administration)
- **Category**: `social_insurance`, `immigration`, `disability_rights`, `government_form`
- **Content**: Federal government forms, information brochures, legal documents
- **Images**: First-page renders of government PDFs
- **Languages**: Primarily German, some French/Italian

### 4. Swiss News
- **Category**: `news`
- **Sources**: swissinfo.ch (English), nzz.ch (German)
- **Content**: News articles on Swiss politics, economy, culture, science
- **Images**: Article header images

### 5. Swiss Product Catalogs
- **Category**: `product`
- **Sources**: Migros (migros.ch), Coop (coop.ch)
- **Content**: Product names, prices (CHF), brands, categories
- **Images**: Product photos

## Usage

### Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Collecting the Dataset

```bash
# Run all scrapers (target ~800 examples)
python collect.py

# Run specific scrapers
python collect.py --scrapers sbb zvv

# Set custom limit
python collect.py --limit 500

# Custom output directory
python collect.py --output-dir ./my_dataset
```

### Loading the Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("swiss_dataset/dataset")

# Access splits
train_ds = ds["train"]
val_ds = ds["val"]
test_ds = ds["test"]

# View an example
example = train_ds[0]
print(example["text"])
print(example["qa_pairs"])
# image is loaded as PIL Image
example["image"].show()
```

### Example Record

```python
{
    "image": <PIL.PngImagePlugin.PngImageFile>,
    "text": "Station: Zürich HB\nStation ID: 8503000\nCoordinates: 8.5403, 47.3778",
    "language": "de",
    "source": "sbb",
    "category": "transport",
    "qa_pairs": [
        {"question": "What is the station ID of Zürich HB?", "answer": "8503000"},
        {"question": "Where is Zürich HB located?", "answer": "Coordinates: 8.5403, 47.3778"}
    ],
    "metadata": "{\"station\": \"Zürich HB\", \"station_id\": \"8503000\"}"
}
```

## Dataset Card Authors

Generated with `collect.py` from the `vlm-swiss-data` project.

## License

The dataset is composed of publicly available Swiss data:

- **transport.opendata.ch**: Open data, CC BY 4.0
- **admin.ch**: Swiss federal documents, public domain (Swiss federal copyright law Art. 5)
- **Wikimedia Commons**: Various open licenses (CC BY-SA, public domain)
- **swissinfo.ch / nzz.ch**: Scraped content subject to respective terms of use; for research/fine-tuning purposes only
- **migros.ch / coop.ch**: Product data for research purposes; images subject to retailer terms

Users are responsible for compliance with each source's terms of use.

## Citation

```bibtex
@dataset{swiss_vlm_2025,
  title={Swiss Multimodal Document Dataset for VLM Fine-Tuning},
  year={2025},
  note={Multimodal dataset of Swiss transport, government, news, and retail data}
}
```
