import logging
import re
from pathlib import Path

from librosa.util import find_files

from s3prl import Container, cache
from s3prl.util import registry

from .base import Corpus

IEMOCAP_SESSION_NUM = 5
LABEL_DIR_PATH = "dialog/EmoEvaluation"
WAV_DIR_PATH = "sentences/wav"


class IEMOCAP(Corpus):
    """
    Parse the IEMOCAP dataset

    Args:
        dataset_root: (str) The dataset root of IEMOCAP
    """

    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        self.sessions = [
            Container(self._preprocess_single_session(self.dataset_root, session_id))
            for session_id in range(1, IEMOCAP_SESSION_NUM + 1)
        ]

        self._all_data = Container()
        for session in self.sessions:
            self._all_data.add(session.improvised)
            self._all_data.add(session.scripted)

    @staticmethod
    @cache()
    def _preprocess_single_session(dataset_root: Path, session_id: int):
        data = dict(
            improvised={},
            scripted={},
        )

        session_dir = dataset_root / f"Session{session_id}"
        label_dir = session_dir / LABEL_DIR_PATH
        wav_root_dir = session_dir / WAV_DIR_PATH
        wav_paths = find_files(wav_root_dir)
        for wav_path in wav_paths:
            wav_path = Path(wav_path)
            spk_and_act_and_scene = wav_path.parts[-2]
            label_file = label_dir / f"{spk_and_act_and_scene}.txt"
            with label_file.open() as file:
                content = file.read()
            result = re.search(rf"{str(wav_path.stem)}\t(.+)\t", content)
            speaker = spk_and_act_and_scene.split("_")[0]
            act = "improvised" if "impro" in spk_and_act_and_scene else "scripted"
            emotion = result.groups()[0]
            unique_id = wav_path.stem

            data[act][unique_id] = dict(
                wav_path=str(wav_path),
                speaker=speaker,
                act=act,
                emotion=emotion,
                session_id=session_id,
            )

        return data

    @property
    def all_data(self):
        """
        Return:
            :obj:`s3prl.base.container.Container`

            all the data points of IEMOCAP in the format of

            .. code-block:: yaml

                data_id1:
                    wav_path (str): The waveform path
                    speaker (str): The speaker name
                    act (str): improvised / scripted
                    emotion (str): The emotion label
                    session_id (int): The session

                data_id2:
                    ...
        """
        return Container(self._all_data)

    def get_whole_session(self, session_id: int):
        """
        Args:
            session_id (int): The session index selected from 1, 2, 3, 4, 5

        Return:
            :obj:`s3prl.base.container.Container`

            data points in a single session (containing improvised and scripted recordings) in the
            same format as :obj:`all_data`
        """
        output = Container()
        output.add(self.get_session_with_act(session_id, "improvised"))
        output.add(self.get_session_with_act(session_id, "scripted"))
        return Container(output)

    def get_session_with_act(self, session_id: int, act: str):
        """
        Args:
            session_id (int): The session index selected from 1, 2, 3, 4, 5
            act (str): 'improvised' or 'scripted'

        Return:
            :obj:`s3prl.base.container.Container`

            data points in a single session with a specific act (either improvised or scripted) in the
            same format as :obj:`all_data`
        """
        assert act in ["improvised", "scripted"]
        return Container(self.sessions[session_id - 1][act])

    @classmethod
    def download_dataset(cls, tgt_dir: str) -> None:
        import os
        import tarfile

        import requests

        assert os.path.exists(
            os.path.abspath(tgt_dir)
        ), "Target directory does not exist"

        def unzip_targz_then_delete(filepath: str):
            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(tgt_dir))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logging.info(f"Saving {filename} to", os.path.abspath(filepath))
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logging.info(f"{filename} successfully downloaded")
                unzip_targz_then_delete(filepath)
            else:
                logging.info(f"Download failed: status code {r.status_code}\n{r.text}")

        if not os.path.exists(os.path.join(os.path.abspath(tgt_dir), "IEMOCAP/")):
            download_from_url("http://140.112.21.28:9000/IEMOCAP.tar.gz")
        logging.info(
            f"IEMOCAP dataset downloaded. Located at {os.path.abspath(tgt_dir)}/IEMOCAP/"
        )


@registry.put()
def iemocap_for_superb(dataset_root: str, test_fold: int = 1, n_jobs: int = 4):
    """
    This is the specific setting used in the SUPERB paper, where we only use
    4 emotion classes: :code:`happy`, :code:`angry`, :code:`neutral`, and :code:`sad`
    with balanced data points and the :code:`excited` class is merged into :code:`happy` class.

    When test_fold is 0, then session 1 will be used as testing set; session 2 will be
    used as validation set; the others are used as the training set

    Args:
        dataset_root (str): The dataset root of IEMOCAP
        test_fold (int): the fold for cross validation, starting from index 0

    Return:
        :obj:`s3prl.base.container.Container`

        .. code-block:: yaml

            train_data:
                data_id1:
                    wav_path (str): The waveform path
                    label (str): The emotion label

                data_id2:
                    ...

            valid_data:
                same format as train_data

            test_data:
                same format as train_data
    """

    def format_fields(data_points):
        return {
            key: dict(
                wav_path=value.wav_path,
                label=value.emotion,
            )
            for key, value in data_points.items()
        }

    def filter_data(data: Container):
        for key in list(data.keys()):
            data_point = data[key]
            if data_point.emotion not in ["neu", "hap", "ang", "sad", "exc"]:
                del data[key]
            if data_point.emotion == "exc":
                data_point.emotion = "hap"
        return data

    corpus = IEMOCAP(dataset_root, n_jobs)
    test_session = test_fold + 1
    valid_session = (test_session + 1) % IEMOCAP_SESSION_NUM
    train_sessions = [
        s + 1
        for s in list(range(IEMOCAP_SESSION_NUM))
        if s + 1 not in [valid_session, test_session]
    ]
    train_data = Container()
    for session_id in train_sessions:
        train_data.add(corpus.get_whole_session(session_id))
    valid_data = corpus.get_whole_session(valid_session)
    test_data = corpus.get_whole_session(test_session)

    return Container(
        train_data=format_fields(filter_data(train_data)),
        valid_data=format_fields(filter_data(valid_data)),
        test_data=format_fields(filter_data(test_data)),
    )
