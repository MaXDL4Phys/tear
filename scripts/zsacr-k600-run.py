import os
# Description: Script to run the zsacr-ucf101 experiment with different configurations

# Define values we want to iterate over
archs=['ViT-B/16', 'ViT-B/32']
prompts = ['only_label', 'description', 'decomposition', 'context', 'all']
models = ['simple'] # only simple in interesting for our work #, 'ensemble']
templates = ['True', 'False']
conditionings = ['True', 'False']
n_frames = [16, 32]

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
                    for fps in n_frames:
                        print(f"Frames: {fps}")
                        command = f"""
                        CUDA_VISIBLE_DEVICES=0 \
                        HYDRA_FULL_ERROR=1 \
                        WANDB_DISABLE_SERVICE=True \
                        python -m src.eval  experiment="k600" \
                        model.decomposition.alpha=0.8 \
                        model.network.arch={arch} \
                        model.network.temperature=0.5 \
                        logger.wandb.offline=False \
                        logger.wandb.tags=["t_1"] \
                        logger.wandb.project="zsacr-k600-{name}" \
                        logger.wandb.name={name}-{prompt}-{template}-{conditioning} \
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
