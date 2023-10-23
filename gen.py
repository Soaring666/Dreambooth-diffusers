import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained("/home/flyvideo/PCH/diffusion/stable-diffusion/stable-diffusion-v1-5", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("/home/flyvideo/PCH/diffusion/stable-diffusion/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("/home/flyvideo/PCH/diffusion/stable-diffusion/stable-diffusion-v1-5", subfolder="text_encoder")
# unet = UNet2DConditionModel.from_pretrained("/home/flyvideo/PCH/diffusion/stable-diffusion/stable-diffusion-v1-5", subfolder="unet")
unet = UNet2DConditionModel.from_pretrained("/home/flyvideo/PCH/diffusion/Dreambooth-diffusers/dreambooth-output/dog", subfolder="unet")

from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_pretrained("/home/flyvideo/PCH/diffusion/stable-diffusion/stable-diffusion-v1-5", subfolder="scheduler")

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

def gen(
    now_epoch: int,
    batch_size: int,
    save_dir: str,
    prompt,
):
    batch_size = batch_size
    prompt = prompt * batch_size
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 50  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(now_epoch)  # Seed generator to create the inital latent noise

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    for i in range(batch_size):
        with torch.no_grad():
            image = vae.decode(latents[i].unsqueeze(0)).sample
            
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        # pil_images = [Image.fromarray(image) for image in images]
        pil_images = Image.fromarray(image)

        import torchvision.transforms as T
        # T.ToPILImage()
        os.makedirs(save_dir, exist_ok=True)
        pil_images.save(f'{save_dir}/%05d.png' % (i + now_epoch * batch_size))
        # for i in range(len(pil_images)):
        #     pil_images[i].save(f'{save_dir}/%05d.png' % (i + now_epoch * batch_size))

if __name__ == "__main__":
    epoch = 1
    batch_size = 5
    # save_dir = "data/class_dir/cat"
    save_dir = "generate"
    # prompt = ["a photo of a cat"]
    prompt = ["a sks cat driving car"]
    for i in range(epoch):
        gen(i, batch_size, save_dir, prompt)
    
    gen_number = epoch * batch_size
    print(f"****** successfully generate {gen_number} images *******")