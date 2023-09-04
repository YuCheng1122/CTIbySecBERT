# SecBERT Project

## Description
This project aims to fine-tune the SecBERT model for Cyber Threat Intelligence (CTI) tasks.It includes various scripts and data for training and evaluating the model.

## Directory Structure

```
CTIbySecBERT/
|-- cti/
|   |-- enterprise-attack/
|   |-- mobile-attack/
|   |-- ... (other directories and files)
|-- data/
|-- scripts/
|-- secbert_env/
|-- .env
|-- .gitignore
|-- main.py
|-- README.md
|-- requirement.txt
```

## Prerequisites
- Python 3.x
- pip

## Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/SecBERT.git
cd SecBERT
# Clone the MITRE CTI repository
$ git clone https://github.com/mitre/cti.git
```

### Create a Virtual Environment
```bash
python3 -m venv secbert_env
```

### Activate the Virtual Environment
- On Windows:
  ```bash
  secbert_env\Scripts\activate
  ```
  
- On macOS and Linux:
  ```bash
  source secbert_env/bin/activate
  ```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root directory and add your VirusTotal API key:
```env
VIRUSTOTAL_API_KEY=your_api_key_here
```

### PyTorch Installation

- **Windows without GPU:**
  ```bash
  pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- **Windows with NVIDIA GPU:**
  ```bash
  pip install torch torchvision torchaudio
  ```

- **Linux or macOS:**
  ```bash
  pip install torch torchvision torchaudio
  ```

## Usage
To collect domain data using VirusTotal, navigate to the `scripts/dataCollection` directory and run `virusTotal.py`:

```bash
cd scripts/dataCollection
python virusTotal.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.