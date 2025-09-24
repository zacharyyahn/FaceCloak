
from DiffAM.utils.align_utils import run_alignment
from PIL import Image
import numpy as np
import torch
import torchvision.utils as tvu
from DiffAM.utils.diffusion_utils import get_beta_schedule, denoising_step
import os
from tqdm import tqdm

def amtgan_makeup(path, inference, postprocess):
    reference_path = "src/AMTGAN/assets/datasets/MT-dataset/images/makeup/0d384dbbcc121ca5049c423f81c26e6a.png"
    source = Image.open(path)
    reference = Image.open(reference_path)
    try:
        image, face = inference.transfer(source, reference, with_face=True)
    except Exception as e:
        print("Encountered error:", e)
        return None
    # source_crop = source.crop((face.left(), face.top(), face.right(), face.bottom()))
    # image = postprocess(source_crop, image)
    im = np.array(image)
    return im


def diffam_makeup(path, model, config, device):
    n = 1
    # try:
    #     img = run_alignment(path, output_size=256)
    # except:
    img = Image.open(path).convert("RGB")

    img = img.resize((256,256))
    img = np.array(img)/255
    img = torch.from_numpy(img).type(torch.FloatTensor).permute(
        2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
    img = img.to(device)
    # tvu.save_image(img, os.path.join(
    #     self.args.image_folder, f'0_orig.png'))
    x0 = (img - 0.5) * 2.

    with torch.no_grad():
        # ---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
        n_inv_step = 20
        t_0 = 60
        seq_inv = np.linspace(
            0, 1, n_inv_step) * t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        x = x0.clone()
        with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
            for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                t = (torch.ones(n) * i).to(device)
                t_prev = (torch.ones(n) * j).to(device)

                betas_orig = get_beta_schedule(
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
                )
                betas = torch.from_numpy(betas_orig).float().to(device)

                alphas = 1.0 - betas_orig
                alphas_cumprod = np.cumprod(alphas, axis=0)
                alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
                posterior_variance = betas_orig * \
                (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

                logvar = np.log(np.append(posterior_variance[1], betas_orig[1:]))

                x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                    logvars=logvar,
                                    sampling_type='ddim',
                                    b=betas,
                                    eta=0,
                                    learn_sigma=False,
                                    ratio=0,
                                    )

                progress_bar.update(1)
                x_lat = x.clone()

        # ----------- Generative Process -----------#
        # print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
        #         f" Steps: {self.args.n_test_step}/{self.args.t_0}")
        n_test_step = 6
        seq_test = np.linspace(
            0, 1, n_test_step) * t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        n_iter = 1
        for it in range(n_iter):
            x = x_lat.clone()
            #tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
            #                                            f'1_lat_ninv{self.args.n_inv_step}.png'))

            with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(n) * i).to(device)
                    t_next = (torch.ones(n) * j).to(device)

                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                        logvars=logvar,
                                        sampling_type="ddim",
                                        b=betas,
                                        eta=0.0,
                                        learn_sigma=False,
                                        ratio=1)

                    # added intermediate step vis
                    # if (i - 99) % 100 == 0:
                    #     tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                    #                                                 f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                    progress_bar.update(1)

            x0 = x.clone()
            tvu.save_image((x+1)*0.5, "/home/hice1/zyahn3/scratch/FacePrivacy/data/cloaked/test_diffam/tvu_im.png")
            out = ((x + 1) * 0.5).squeeze().detach().cpu().numpy()
            out = np.transpose(out, (1, 2, 0)) * 255.0
            out = out.astype(np.uint8)
            return out
            # if self.args.model_path:
            #     tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
            #                                                 f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png"))
            # else:
            #     tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
            #                                                 f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))
