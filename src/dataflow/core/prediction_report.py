import abc
import csv
from typing import Dict, List, Sequence, Union

import pandas as pd


class PredictionReportDatum(abc.ABC):
    @abc.abstractmethod
    def flatten(self) -> Dict[str, Union[str, int]]:
        raise NotImplementedError()


def save_prediction_report_tsv(
    prediction_report: Sequence[PredictionReportDatum], prediction_report_tsv: str,
) -> None:
    """Converts prediction results into a pandas dataframe and saves it a tsv report file.
    """
    prediction_report_df = pd.DataFrame(
        [datum.flatten() for datum in prediction_report]
    )
    prediction_report_df.to_csv(
        prediction_report_tsv,
        sep="\t",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )


def save_prediction_report_txt(
    prediction_report: Sequence[PredictionReportDatum],
    prediction_report_txt: str,
    field_names: List[str],
) -> None:
    """Prints prediction results into an easy-to-read text report file."""
    with open(prediction_report_txt, "w") as fp:
        for datum in prediction_report:
            fp.write("=" * 16)
            fp.write("\n")

            flatten_fields = datum.flatten()
            for field_name in field_names:
                field_value = flatten_fields[field_name]
                # use "hypo" not "prediction" as the name here just to make it visually aligned with "gold"
                if field_name == "prediction":
                    field_name = "hypo"
                print(f"{field_name}\t{field_value}", file=fp)
