import torch
import time
from typing import Union
from io import BytesIO
from .utils.tools import prepare_text
from scipy.io.wavfile import write as wav_write


# noinspection PyUnusedLocal
class GladosTTS:
    def __init__(self, *args, **kwargs):
        print("Initializing TTS Engine...")
        # Select the device
        if torch.is_vulkan_available():
            self.device = 'vulkan'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # Load models
        # TODO: relative path
        self.glados = torch.jit.load('/home/giang/Projects/Salieri/mouth/GladosTTS/models/glados.pt')
        self.vocoder = torch.jit.load('/home/giang/Projects/Salieri/mouth/GladosTTS/models/vocoder-cpu-hq.pt',
                                      map_location=self.device)
        # Prepare models in RAM
        for i in range(2):
            init = self.glados.generate_jit(prepare_text(str(i)))
            init_mel = init['mel_post'].to(self.device)
            init_vo = self.vocoder(init_mel)

    def text_to_speech(self, text: str = None,
                       emotion: list = None,
                       output_to_file=False,
                       *args, **kwargs) -> Union[None, BytesIO]:
        if not text:
            return None
        x = prepare_text(text).to('cpu')

        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados.generate_jit(x)
            print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            old_time = time.time()
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')

            if output_to_file:
                # Write audio file to disk
                # 22,05 kHz sample rate
                output_file = BytesIO()
                wav_write(output_file, 22050, audio)
                audio = output_file
        return audio
