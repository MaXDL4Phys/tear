import os

# Define values we want to iterate over
archs=['ViT-B/16', 'ViT-B/16']
prompts = ['context', 'decomposition', 'all','description', 'decomposition', 'context', 'situation',]
models = ['simple', 'ensemble']
templates = ['True', 'False']
conditionings = ['True', 'False']

# Iterate over the values and run the appropriate command
for arch in archs:
    print(f"Arch: {arch}")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        for model in models:
            print(f"Model: {model}")
            for template in templates:
                print(f"Template: {template}")
                for conditioning in conditionings:
                    print(f"Conditioning: {conditioning}")
                    name = 'ViTb32' if arch == 'ViT-B/32' else 'ViTb16'
                    print(f"Conditioning: {conditioning}")
                    command = f"""
                    CUDA_VISIBLE_DEVICES=0 \
                    HYDRA_FULL_ERROR=1 \
                    python -m src.eval  experiment="hmdb51" \
                    model.decomposition.alpha=0.8 \
                    model.network.arch={arch} \
                    model.network.temperature=0.5 \
                    logger.wandb.offline=False \
                    logger.wandb.tags=["t_1"] \
                    logger.wandb.project="zsacr-hmdb-llama-{name}" \
                    logger.wandb.name={name}-{prompt}-{model}-{template}-{conditioning} \
                    trainer=gpu \
                    trainer.devices=1 \
                    data.num_workers=16 \
                    model.decomposition.use_templates={template} \
                    model.decomposition.prompts={prompt} \
                    model.decomposition.input_conditioning={conditioning} \
                    model.method={model} \
                    data.batch_size=16 
                    """
                    os.system(command)
