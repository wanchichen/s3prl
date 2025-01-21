import logging

import torch
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)

try:
    from espnet2.tasks.hubert import HubertTask
    from espnet2.tasks.ssl import SSLTask
except ModuleNotFoundError:
    HubertTask = None
    logger.warning("ESPnet is not installed, cannot use espnet_hubert upstream")


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, config=None, use_torchaudio=False, **kwargs):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert (
            HubertTask is not None
        ), "ESPnet is not installed, run `external_tools/install_espnet.sh` to install"
        
        self.use_torchaudio = use_torchaudio

        if use_torchaudio:
            hubert_model, hubert_train_args = HubertTask.build_model_from_file(
                config,
                ckpt,
                device,
            )
            self.model = hubert_model.encoder.hubert_pretrain_model
        else:
            hubert_model, hubert_train_args = SSLTask.build_model_from_file(
                config,
                ckpt,
                device,
            )
            self.model = hubert_model

        self.device = next(hubert_model.parameters()).device

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(self.device)
        wavs = pad_sequence(wavs, batch_first=True).to(self.device)
        if self.use_torchaudio:
            feats = self.model.wav2vec2.extract_features(wavs, wav_lengths)[
                0
            ]  # (time, feat_dim))
        else:
            feats, mask_info, pen = self.model.encode(wavs, wav_lengths, use_mask=self.training, use_final_output=False)

        return {"hidden_states": feats}
