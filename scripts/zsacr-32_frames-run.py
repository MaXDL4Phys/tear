import os
# Description: Script to run the zsacr-ucf101 experiment with different configurations

# Define values we want to iterate over
experiments = [ 'k600_splitted'] #'ucf101', 'hmdb51',
archs=['ViT-B/16']# , 'ViT-B/32']
prompts = [ 'all'] #'only_label', 'description', 'decomposition', 'context',
models = ['simple'] # only simple in interesting for our work #, 'ensemble']
templates = ['True'] #, 'False']
conditionings = ['True'] #, 'False']
n_frames = [32]
splits = [ 1]
# Iterate over the values and run the appropriate command
for exp in experiments:
    print(f"Experiment: {exp}")
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
                        for fps in n_frames:
                            print(f"Frames: {fps}")
                            for split in splits:
                                print(f"Split: {split}")
                                name = 'ViTb32' if arch == 'ViT-B/32' else 'ViTb16'
                                command = f"""
                                CUDA_VISIBLE_DEVICES=0 \
                                HYDRA_FULL_ERROR=1 \
                                WANDB_DISABLE_SERVICE=True \
                                python -m src.eval  \
                                experiment={exp} \
                                model.decomposition.alpha=0.8 \
                                model.network.arch={arch} \
                                model.network.temperature=0.5 \
                                logger.wandb.offline=False \
                                logger.wandb.tags=["t_1"] \
                                logger.wandb.project="zsar-{exp}-{name}" \
                                logger.wandb.name={exp}-{name}-{prompt}-{template}-{conditioning}-{split} \
                                trainer=gpu \
                                trainer.devices=1 \
                                data.split={split} \
                                data.num_workers=16 \
                                data.n_frames={fps} \
                                model.decomposition.use_templates={template} \
                                model.decomposition.prompts={prompt} \
                                model.decomposition.input_conditioning={conditioning} \
                                model.method={model} \
                                data.batch_size=8
                                """
                                os.system(command)
