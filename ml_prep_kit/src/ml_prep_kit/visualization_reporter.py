"""Utilitários simples para relatórios visuais em notebooks."""

import pandas as pd


class VisualizationReporter:
    """Cria tabelas e gráficos simples para análise gerencial.

    Use esta classe em notebooks quando precisar apresentar KPIs, rankings,
    comparações entre valores atuais e esperados, ou evolução ao longo do
    tempo. Os métodos usam pandas e matplotlib.

    Exemplo:
        reporter = VisualizationReporter()

        kpis = reporter.create_kpi_table([
            {
                "label": "Usuários",
                "value": "3.000",
                "note": "Amostra analisada.",
            },
        ])

        reporter.plot_comparison_bar(
            df=projection,
            label_column="scenario",
            current_column="current_value",
            expected_column="expected_value",
            title="Atual vs esperado",
            xlabel="Valor estimado",
        )
    """

    def create_kpi_table(
        self,
        cards: list[dict[str, str]],
    ) -> pd.DataFrame:
        """Retorna os KPIs em formato de tabela.

        Exemplo:
            kpis = reporter.create_kpi_table([
                {
                    "label": "Pedidos",
                    "value": "10.000",
                    "note": "Pedidos históricos.",
                },
            ])
        """
        return pd.DataFrame(cards)

    def plot_comparison_bar(
        self,
        df: pd.DataFrame,
        label_column: str,
        current_column: str,
        expected_column: str,
        title: str,
        xlabel: str,
    ) -> None:
        """Compara valor atual e esperado em barras horizontais.

        Exemplo:
            reporter.plot_comparison_bar(
                df=projection,
                label_column="department",
                current_column="current_value",
                expected_column="expected_value",
                title="Consumo atual vs esperado",
                xlabel="Valor estimado",
            )
        """
        chart = df.sort_values(current_column, ascending=True)
        plt = self._get_pyplot()

        ax = chart.plot.barh(
            x=label_column,
            y=[current_column, expected_column],
            figsize=(10, max(4, len(chart) * 0.45)),
            color=["#64748b", "#16a34a"],
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.legend(["Atual", "Esperado"])
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_ranking_bar(
        self,
        df: pd.DataFrame,
        label_column: str,
        value_column: str,
        title: str,
        xlabel: str,
    ) -> None:
        """Exibe um ranking simples em barras horizontais.

        Exemplo:
            reporter.plot_ranking_bar(
                df=class_metrics,
                label_column="class_name",
                value_column="f1",
                title="F1 por classe",
                xlabel="F1",
            )
        """
        chart = df.sort_values(value_column, ascending=True)
        plt = self._get_pyplot()

        ax = chart.plot.barh(
            x=label_column,
            y=value_column,
            figsize=(10, max(4, len(chart) * 0.45)),
            color="#2563eb",
            legend=False,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_timeline(
        self,
        df: pd.DataFrame,
        x_column: str,
        current_column: str,
        expected_column: str,
        title: str,
        ylabel: str,
    ) -> None:
        """Compara valor atual e esperado ao longo do tempo.

        Exemplo:
            reporter.plot_timeline(
                df=timeline,
                x_column="period",
                current_column="current_value",
                expected_column="expected_value",
                title="Evolução do consumo",
                ylabel="Valor estimado",
            )
        """
        plt = self._get_pyplot()

        ax = df.plot(
            x=x_column,
            y=[current_column, expected_column],
            marker="o",
            figsize=(10, 4),
            color=["#64748b", "#16a34a"],
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.legend(["Atual", "Esperado"])
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        row_column: str,
        column_column: str,
        value_column: str,
        title: str,
        colorbar_label: str,
    ) -> None:
        """Exibe uma matriz de intensidade entre duas dimensões.

        Exemplo:
            reporter.plot_heatmap(
                df=associations,
                row_column="source",
                column_column="target",
                value_column="orders",
                title="Categorias compradas juntas",
                colorbar_label="Pedidos",
            )
        """
        matrix = df.pivot(
            index=row_column,
            columns=column_column,
            values=value_column,
        ).fillna(0)

        plt = self._get_pyplot()
        fig, ax = plt.subplots(
            figsize=(
                max(8, len(matrix.columns) * 0.8),
                max(5, len(matrix.index) * 0.6),
            )
        )
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")

        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(matrix.index)

        for row_index, row_name in enumerate(matrix.index):
            for column_index, column_name in enumerate(matrix.columns):
                value = matrix.loc[row_name, column_name]
                if value <= 0:
                    continue

                ax.text(
                    column_index,
                    row_index,
                    f"{value:,.0f}",
                    ha="center",
                    va="center",
                    color="#111827",
                    fontsize=8,
                )

        fig.colorbar(image, ax=ax, label=colorbar_label)
        plt.tight_layout()
        self._show_or_close(plt)

    def _get_pyplot(self):
        """Importa matplotlib somente quando um gráfico é necessário."""
        from matplotlib import pyplot as plt

        return plt

    def _show_or_close(self, plt) -> None:
        """Exibe o gráfico no notebook ou fecha em backend não interativo."""
        if plt.get_backend().lower() == "agg":
            plt.close()
            return

        plt.show()
