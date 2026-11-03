from setuptools import setup, find_packages

setup(
    name="lora_fai",
    version="0.1.0",
    description="Efficient Adaptation of Large Language Models: A From-Scratch Implementation and Analysis of LoRA and QLoRA",
    author="LoRA Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "evaluate>=0.4.1",
        "bitsandbytes>=0.43.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.1",
        "gradio>=4.26.0",
    ],
)
