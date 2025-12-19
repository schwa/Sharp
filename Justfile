# SHARP Swift - Model Setup

set shell := ["bash", "-cu"]

model_url := "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
model_dir := "Models"
pytorch_model := model_dir / "sharp_2572gikvuh.pt"
mlpackage := model_dir / "SharpPredictor.mlpackage"
mlmodelc := model_dir / "SharpPredictor.mlmodelc"

# Default: show available commands
default:
    @just --list

# Download PyTorch model from Apple CDN
download-model:
    @mkdir -p {{model_dir}}
    @if [ -f "{{pytorch_model}}" ]; then \
        echo "✅ Model already exists: {{pytorch_model}}"; \
    else \
        echo "📥 Downloading SHARP model (2.6GB)..."; \
        curl -L --progress-bar "{{model_url}}" -o "{{pytorch_model}}"; \
        echo "✅ Downloaded to {{pytorch_model}}"; \
    fi

# Convert PyTorch model to CoreML .mlpackage
# Requires ml-sharp: git clone https://github.com/apple/ml-sharp.git
convert-model ml_sharp_path: download-model
    @if [ ! -d "{{ml_sharp_path}}/src/sharp" ]; then \
        echo "❌ ml-sharp not found at {{ml_sharp_path}}"; \
        echo ""; \
        echo "Clone Apple's ml-sharp repo first:"; \
        echo "  git clone https://github.com/apple/ml-sharp.git {{ml_sharp_path}}"; \
        exit 1; \
    fi
    @if [ -d "{{mlpackage}}" ]; then \
        echo "✅ CoreML model already exists: {{mlpackage}}"; \
    else \
        echo "🔄 Converting PyTorch → CoreML..."; \
        PYTHONPATH={{ml_sharp_path}}/src uv run \
            --with torch --with torchvision --with timm \
            --with coremltools --with plyfile --with scipy \
            scripts/export_coreml.py; \
    fi

# Compile .mlpackage to .mlmodelc (faster loading)
compile-model: 
    @if [ ! -d "{{mlpackage}}" ]; then \
        echo "❌ No .mlpackage found. Run 'just convert-model <path-to-ml-sharp>' first"; \
        exit 1; \
    fi
    @if [ -d "{{mlmodelc}}" ]; then \
        echo "✅ Compiled model already exists: {{mlmodelc}}"; \
    else \
        echo "🔧 Compiling CoreML model..."; \
        xcrun coremlcompiler compile "{{mlpackage}}" "{{model_dir}}"; \
        echo "✅ Compiled to {{mlmodelc}}"; \
    fi

# Full setup: download, convert, and compile model
setup-model ml_sharp_path: (convert-model ml_sharp_path) compile-model
    @echo "🎉 Model ready!"

# Generate PLY from input image using SHARP model
generate-ply input output=".":
    swift run sharp-cli -i "{{input}}" -o "{{output}}" -m "{{mlmodelc}}"

# Generate PNG screenshot from PLY splat file
screenshot ply output:
    spark-screenshot --splat "{{ply}}" --output "{{output}}" --camera-position 0,0,-1.5
