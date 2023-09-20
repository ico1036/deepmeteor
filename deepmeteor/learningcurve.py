from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict
from pathlib import Path
import typing
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from deepmeteor.result import TrainingResult
from deepmeteor.result import EvaluationResult
from deepmeteor.result import Summary
from deepmeteor.const import get_label


class LearningCurve(list):

    @property
    def df(self):
        return pd.DataFrame(self)


@dataclass
class Monitor:
    training: LearningCurve = field(default_factory=LearningCurve)
    validation: LearningCurve = field(default_factory=LearningCurve)
    epoch: LearningCurve = field(default_factory=LearningCurve)

    @property
    def last_step(self) -> int:
        return len(self.training)

    def to_csv(self, output_dir: Path):
        for key, value in vars(self).items():
            pd.DataFrame(value).to_csv(output_dir / f'{key}.csv')

    def to_json(self, output_dir: Path):
        for key, value in vars(self).items():
            pd.DataFrame(value).to_json(output_dir / f'{key}.json')

    def draw(self,
             name: str,
             output_dir: Path,
             summary: Summary | dict | None = None,
    ) -> None:
        use_train = name in TrainingResult.field_names
        df_train = self.training.df if use_train else None
        if isinstance(summary, Summary):
            summary = asdict(summary)

        label = get_label(name)

        fig = plot_learning_curve(
            name=name,
            label=label,
            df_epoch=self.epoch.df,
            df_train=df_train,
            df_val=self.validation.df,
            summary=summary,
        )
        output_path = output_dir / name
        for suffix in ['.pdf', '.png']:
            fig.savefig(output_path.with_suffix(suffix))
        plt.close(fig)


    def draw_all(self,
                 output_dir: Path,
                 summary: Summary | None = None,
    ):
        for each in EvaluationResult.field_names:
            self.draw(name=each,
                      output_dir=output_dir,
                      summary=summary)


def plot_learning_curve(name: str,
                        label: str,
                        df_epoch: pd.DataFrame,
                        df_train: pd.DataFrame | None,
                        df_val: pd.DataFrame | None,
                        summary: dict | None

):
    fig, ax = plt.subplots()
    fig = typing.cast(plt.Figure, fig)
    ax = typing.cast(plt.Axes, ax)

    if df_train is not None:
        train_x = df_train.index
        train_y = df_train[name]

        train_smooth = lowess(endog=train_y, exog=train_x, frac=0.075, it=0,
                              is_sorted=True)
        train_smooth_x, train_smooth_y = train_smooth.T

        train_style = dict(alpha=0.3, lw=3, color='tab:blue')
        train_smooth_style = dict(lw=3, color='tab:blue')


        _ = ax.plot(train_x, train_y, label='Training', **train_style)
        _ = ax.plot(train_smooth_x, train_smooth_y, label='Training (LOWESS)',
                    **train_smooth_style)

    if df_val is not None:
        val_x = df_epoch.step
        val_y = df_val[name]

        val_style = dict(ls='-', lw=3, color='tab:orange')

        _ = ax.plot(val_x, val_y, label='Validation', **val_style)

    if summary is not None:
        test_x = summary['step']
        test_y = summary['test'][name]
        test_style = dict(s=500, marker='*', color='tab:red')
        ax.scatter(test_x, test_y, label='Test', **test_style)

    xticks = df_epoch.step
    xticklabels = df_epoch.epoch
    if len(xticks) > 10:
        xtick_slicing = slice(None, None, len(xticks) // 5)
        xticks = xticks[xtick_slicing]
        xticklabels = xticklabels[xtick_slicing]
    _ = ax.set_xticks(xticks) # type: ignore
    _ = ax.set_xticklabels(xticklabels) # type: ignore

    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.grid()
    ax.legend()

    fig.tight_layout()
    return fig

