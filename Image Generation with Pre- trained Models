#CODECRAFT_GA_2
Image Generation with
Pre- trained
Models
pip install tensorflow keras_cv --upgrade --quiet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 721.6/721.6 kB 13.5 MB/s eta 0:00:00
import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)
Downloading data from https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true
1356917/1356917 [==============================] - 0s 0us/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5
492466864/492466864 [==============================] - 9s 0us/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5
3439090152/3439090152 [==============================] - 63s 0us/step
50/50 [==============================] - 126s 295ms/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5
198180272/198180272 [==============================] - 2s 0us/step
images = model.text_to_image(
    "cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(images)
50/50 [==============================] - 15s 294ms/step
benchmark_result = []
start = time.time()
images = model.text_to_image(
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Standard", end - start])
plot_images(images)

print(f"Standard model: {(end - start):.2f} seconds")
keras.backend.clear_session()  # Clear session to preserve memory.
50/50 [==============================] - 15s 294ms/step
Standard model: 15.02 seconds
png
keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion()

print("Compute dtype:", model.diffusion_model.compute_dtype)
print(
    "Variable dtype:",
    model.diffusion_model.variable_dtype,
)
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
Compute dtype: float16
Variable dtype: float32
# Warm up model to run graph tracing before benchmarking.
model.text_to_image("warming up the model", batch_size=3)

start = time.time()
images = model.text_to_image(
    "a cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Mixed Precision", end - start])
plot_images(images)

print(f"Mixed precision model: {(end - start):.2f} seconds")
keras.backend.clear_session()
50/50 [==============================] - 24s 229ms/step
50/50 [==============================] - 11s 229ms/step
Mixed precision model: 11.87 seconds
# Set back to the default for benchmarking purposes.
keras.mixed_precision.set_global_policy("float32")

model = keras_cv.models.StableDiffusion(jit_compile=True)
# Before we benchmark the model, we run inference once to make sure the TensorFlow
# graph has already been traced.
images = model.text_to_image("An avocado armchair", batch_size=3)
plot_images(images)
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
50/50 [==============================] - 71s 233ms/step
start = time.time()
images = model.text_to_image(
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["XLA", end - start])
plot_images(images)

print(f"With XLA: {(end - start):.2f} seconds")
keras.backend.clear_session()
50/50 [==============================] - 12s 233ms/step
With XLA: 11.84 seconds
keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(jit_compile=True)
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
# Let's make sure to warm up the model
images = model.text_to_image(
    "Teddy bears conducting machine learning research",
    batch_size=3,
)
plot_images(images)
50/50 [==============================] - 71s 144ms/step
start = time.time()
images = model.text_to_image(
    "A mysterious dark stranger visits the great pyramids of egypt, "
    "high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["XLA + Mixed Precision", end - start])
plot_images(images)

print(f"XLA + mixed precision: {(end - start):.2f} seconds")
50/50 [==============================] - 7s 144ms/step
XLA + mixed precision: 7.51 seconds
print("{:<22} {:<22}".format("Model", "Runtime"))
for result in benchmark_result:
    name, runtime = result
    print("{:<22} {:<22}".format(name, runtime))
Model                  Runtime               
Standard               15.015103816986084    
Mixed Precision        11.867290258407593    
XLA                    11.838508129119873    
XLA + Mixed Precision  7.507506370544434
GPU 0: Tesla T4 (UUID: GPU-83334893-eba1-d1e0-f1ee-bc77eac5b1e7)
Collecting git+https://github.com/robgon-art/dalle-mini.git
  Cloning https://github.com/robgon-art/dalle-mini.git to /tmp/pip-req-build-ei_s85hd
  Running command git clone --filter=blob:none --quiet https://github.com/robgon-art/dalle-mini.git /tmp/pip-req-build-ei_s85hd
  Resolved https://github.com/robgon-art/dalle-mini.git to commit 64592126bd589b8b6f0c3e096ebd3e0b5936fe2e
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (4.47.1)
Requirement already satisfied: einops in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (0.8.0)
Collecting unidecode (from dalle-mini==0.1.1)
  Downloading Unidecode-1.3.8-py3-none-any.whl.metadata (13 kB)
Collecting ftfy (from dalle-mini==0.1.1)
  Downloading ftfy-6.3.1-py3-none-any.whl.metadata (7.3 kB)
Collecting emoji (from dalle-mini==0.1.1)
  Downloading emoji-2.14.0-py3-none-any.whl.metadata (5.7 kB)
Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (11.0.0)
Requirement already satisfied: jax in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (0.4.33)
Requirement already satisfied: flax in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (0.8.5)
Requirement already satisfied: wandb in /usr/local/lib/python3.10/dist-packages (from dalle-mini==0.1.1) (0.19.1)
Requirement already satisfied: numpy>=1.22 in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (1.26.4)
Requirement already satisfied: msgpack in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (1.1.0)
Requirement already satisfied: optax in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (0.2.4)
Requirement already satisfied: orbax-checkpoint in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (0.6.4)
Requirement already satisfied: tensorstore in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (0.1.71)
Requirement already satisfied: rich>=11.1 in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (13.9.4)
Requirement already satisfied: typing-extensions>=4.2 in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (4.12.2)
Requirement already satisfied: PyYAML>=5.4.1 in /usr/local/lib/python3.10/dist-packages (from flax->dalle-mini==0.1.1) (6.0.2)
Requirement already satisfied: jaxlib<=0.4.33,>=0.4.33 in /usr/local/lib/python3.10/dist-packages (from jax->dalle-mini==0.1.1) (0.4.33)
Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax->dalle-mini==0.1.1) (0.4.1)
Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax->dalle-mini==0.1.1) (3.4.0)
Requirement already satisfied: scipy>=1.10 in /usr/local/lib/python3.10/dist-packages (from jax->dalle-mini==0.1.1) (1.13.1)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from ftfy->dalle-mini==0.1.1) (0.2.13)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (3.16.1)
Requirement already satisfied: huggingface-hub<1.0,>=0.24.0 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (0.27.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (24.2)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (2024.11.6)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (2.32.3)
Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (0.21.0)
Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (0.4.5)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers->dalle-mini==0.1.1) (4.67.1)
Requirement already satisfied: click!=8.0.0,>=7.1 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (8.1.7)
Requirement already satisfied: docker-pycreds>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (0.4.0)
Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (3.1.43)
Requirement already satisfied: platformdirs in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (4.3.6)
Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<6,>=3.19.0 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (4.25.5)
Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (5.9.5)
Requirement already satisfied: pydantic<3,>=2.6 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (2.10.3)
Requirement already satisfied: sentry-sdk>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (2.19.2)
Requirement already satisfied: setproctitle in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (1.3.4)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from wandb->dalle-mini==0.1.1) (75.1.0)
Requirement already satisfied: six>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from docker-pycreds>=0.4.0->wandb->dalle-mini==0.1.1) (1.17.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.29,>=1.0.0->wandb->dalle-mini==0.1.1) (4.0.11)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.24.0->transformers->dalle-mini==0.1.1) (2024.10.0)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=2.6->wandb->dalle-mini==0.1.1) (0.7.0)
Requirement already satisfied: pydantic-core==2.27.1 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=2.6->wandb->dalle-mini==0.1.1) (2.27.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->dalle-mini==0.1.1) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->dalle-mini==0.1.1) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->dalle-mini==0.1.1) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->dalle-mini==0.1.1) (2024.12.14)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax->dalle-mini==0.1.1) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax->dalle-mini==0.1.1) (2.18.0)
Requirement already satisfied: absl-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from optax->flax->dalle-mini==0.1.1) (1.4.0)
Requirement already satisfied: chex>=0.1.87 in /usr/local/lib/python3.10/dist-packages (from optax->flax->dalle-mini==0.1.1) (0.1.88)
Requirement already satisfied: etils[epy] in /usr/local/lib/python3.10/dist-packages (from optax->flax->dalle-mini==0.1.1) (1.11.0)
Requirement already satisfied: nest_asyncio in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax->dalle-mini==0.1.1) (1.6.0)
Requirement already satisfied: humanize in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax->dalle-mini==0.1.1) (4.11.0)
Requirement already satisfied: toolz>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from chex>=0.1.87->optax->flax->dalle-mini==0.1.1) (0.12.1)
Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb->dalle-mini==0.1.1) (5.0.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=11.1->flax->dalle-mini==0.1.1) (0.1.2)
Requirement already satisfied: importlib_resources in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax->dalle-mini==0.1.1) (6.4.5)
Requirement already satisfied: zipp in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax->dalle-mini==0.1.1) (3.21.0)
Downloading emoji-2.14.0-py3-none-any.whl (586 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 586.9/586.9 kB 15.4 MB/s eta 0:00:00
Downloading ftfy-6.3.1-py3-none-any.whl (44 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 kB 3.6 MB/s eta 0:00:00
Downloading Unidecode-1.3.8-py3-none-any.whl (235 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 235.5/235.5 kB 20.5 MB/s eta 0:00:00
Building wheels for collected packages: dalle-mini
  Building wheel for dalle-mini (pyproject.toml) ... done
  Created wheel for dalle-mini: filename=dalle_mini-0.1.1-py3-none-any.whl size=34185 sha256=ff6c16abdffb77f9800ea5ccaffc6baa68ee829c87e5265d2f9924e01a5a48dd
  Stored in directory: /tmp/pip-ephem-wheel-cache-q8_x45dl/wheels/57/77/f7/479cb620847c2e5b8208a41180d85072682c1920a24327c810
Successfully built dalle-mini
Installing collected packages: unidecode, ftfy, emoji, dalle-mini
Successfully installed dalle-mini-0.1.1 emoji-2.14.0 ftfy-6.3.1 unidecode-1.3.8
  Preparing metadata (setup.py) ... done
  Building wheel for vqgan-jax (setup.py) ... done
#@title Choose DALL-E Model

choose_model = "DALL-E Mega Full" #@param ["DALL-E Mini", "DALL-E Mega", "DALL-E Mega Full"]

models = {"DALL-E Mini":"dalle-mini/dalle-mini/mini-1:v0",
          "DALL-E Mega":"dalle-mini/dalle-mini/mega-1-fp16:latest",
          "DALL-E Mega Full":"dalle-mini/dalle-mini/mega-1:latest"}

DALLE_COMMIT_ID = None

DALLE_MODEL = models[choose_model]

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

from functools import partial

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-2-ee5b00931e0a> in <cell line: 18>()
     16 
     17 # Load models & tokenizer
---> 18 from dalle_mini import DalleBart, DalleBartProcessor
     19 from vqgan_jax.modeling_flax_vqgan import VQModel
     20 from transformers import CLIPProcessor, FlaxCLIPModel
/usr/local/lib/python3.10/dist-packages/dalle_mini/model/modeling.py in <module>
     31 from jax import custom_jvp, lax
     32 from jax.random import PRNGKey
---> 33 from transformers.generation_flax_utils import FlaxSampleOutput
     34 from transformers.modeling_flax_outputs import (
     35     FlaxBaseModelOutput,

ModuleNotFoundError: No module named 'transformers.generation_flax_utils'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------
!pip install matplotlib-venn
Requirement already satisfied: matplotlib-venn in /usr/local/lib/python3.10/dist-packages (1.1.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (3.8.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (1.26.4)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (1.13.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (4.55.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (1.4.7)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (24.2)
Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (11.0.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (3.2.0)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (2.8.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib->matplotlib-venn) (1.17.0)
!apt-get -qq install -y libfluidsynth1
E: Package 'libfluidsynth1' has no installation candidate
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
Collecting libarchive
  Using cached libarchive-0.4.7.tar.gz (23 kB)
  Preparing metadata (setup.py) ... done
Collecting nose (from libarchive)
  Using cached nose-1.3.7-py3-none-any.whl.metadata (1.7 kB)
Using cached nose-1.3.7-py3-none-any.whl (154 kB)
Building wheels for collected packages: libarchive
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  Building wheel for libarchive (setup.py) ... error
  ERROR: Failed building wheel for libarchive
  Running setup.py clean for libarchive
Failed to build libarchive
ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (libarchive)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-9-5bee6841787f> in <cell line: 3>()
      1 # https://pypi.python.org/pypi/libarchive
      2 get_ipython().system('apt-get -qq install -y libarchive-dev && pip install -U libarchive')
----> 3 import libarchive

ModuleNotFoundError: No module named 'libarchive'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To install libarchive, click the button below.
---------------------------------------------------------------------------
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot
Requirement already satisfied: pydot in /usr/local/lib/python3.10/dist-packages (3.0.3)
Requirement already satisfied: pyparsing>=3.0.9 in /usr/local/lib/python3.10/dist-packages (from pydot) (3.2.0)
!pip install cartopy
import cartopy
Requirement already satisfied: cartopy in /usr/local/lib/python3.10/dist-packages (0.24.1)
Requirement already satisfied: numpy>=1.23 in /usr/local/lib/python3.10/dist-packages (from cartopy) (1.26.4)
Requirement already satisfied: matplotlib>=3.6 in /usr/local/lib/python3.10/dist-packages (from cartopy) (3.8.0)
Requirement already satisfied: shapely>=1.8 in /usr/local/lib/python3.10/dist-packages (from cartopy) (2.0.6)
Requirement already satisfied: packaging>=21 in /usr/local/lib/python3.10/dist-packages (from cartopy) (24.2)
Requirement already satisfied: pyshp>=2.3 in /usr/local/lib/python3.10/dist-packages (from cartopy) (2.3.1)
Requirement already satisfied: pyproj>=3.3.1 in /usr/local/lib/python3.10/dist-packages (from cartopy) (3.7.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (4.55.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (1.4.7)
Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (11.0.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (3.2.0)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (2.8.2)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from pyproj>=3.3.1->cartopy) (2024.12.14)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.6->cartopy) (1.17.0)
#@title Generate Images

import matplotlib.pyplot as plt
import random

prompt = 'a painting of rolling farmland' #@param {type:"string"}
prompts = [prompt]
tokenized_prompts = processor(prompts)
tokenized_prompt = replicate(tokenized_prompts)

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

# number of predictions per prompt
n_predictions = 7

# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

# generate images
images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):
    # get a new key
    key, subkey = jax.random.split(key)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)
        # display(img)
        # print()


cols = n_predictions
index = 0
fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(cols*4, 4))
for i in range(cols):
  axes[i].axis("off")
  axes[i].set_title(str(index+1), fontsize=15)  
  axes[i].imshow(images[index])
  index += 1

fig.tight_layout()
plt.show()
print(prompt)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-3-8e57f0e5cff5> in <cell line: 8>()
      6 prompt = 'a painting of rolling farmland' #@param {type:"string"}
      7 prompts = [prompt]
----> 8 tokenized_prompts = processor(prompts)
      9 tokenized_prompt = replicate(tokenized_prompts)
     10 

NameError: name 'processor' is not defined
#@title Show Larger Image

choose = 6 #@param {type:"slider", min:1, max:7, step:1}
from IPython.display import Image
images[choose-1].save("output.png")
Image('output.png')
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-4-f34c4e607bab> in <cell line: 5>()
      3 choose = 6 #@param {type:"slider", min:1, max:7, step:1}
      4 from IPython.display import Image
----> 5 images[choose-1].save("output.png")
      6 Image('output.png')

NameError: name 'images' is not defined
