import abc
from re import Match
from typing import Any, Optional, Tuple, Union
import pandas as pd
import sqlite3

from logger_factory import get_logger
from utils import RecoverableError, TableError


class ITableStrategy(abc.ABC):
    @abc.abstractmethod
    def operation(self, table: Any, operation: str, args: Match[str], table_name: str) -> Any:
        pass

    @abc.abstractmethod
    def get_table_strategy(self) -> str:
        pass

    @abc.abstractmethod
    def json_to_table(self, json_data) -> Tuple[Any, str]:
        pass

    @abc.abstractmethod
    def table_to_str(self, table: Any, table_caption: Optional[str]) -> str:
        pass

    @abc.abstractmethod
    def is_compatible_table(self, table: Any) -> bool:
        pass


class Table:
    _strategy: ITableStrategy

    def __init__(self, strategy: ITableStrategy):
        if not isinstance(strategy, ITableStrategy):
            raise TableError("Provided strategy must be an instance of ITableStrategy.")
        self._strategy = strategy
        self._table: Union[pd.DataFrame, sqlite3.Connection, None] = None
        self._caption: Optional[str]

    def perform_operation(self, operation: str, args: Match[str], table_name: str):
        table_logger = get_logger(table_name, "table")
        try:
            if self._table is None:
                raise TableError("No data assigned to table")
            self._table = self._strategy.operation(
                self._table, operation, args, table_name
            )
            table_logger.debug(f"Performed operation: {operation} with args: {args}")
        except RecoverableError:
            raise
        except Exception as e:
            raise TableError(
                f"[perform_operation] - Failed to perform operation '{operation}': {e}"
            ) from e

    def load_from_json(self, json_data):
        table_logger = get_logger("GLOBAL", "table")
        try:
            self._table, self._caption = self._strategy.json_to_table(json_data)
            table_logger.debug(f"Loaded table: {self._caption}")
        except Exception as e:
            raise TableError(f"[load_from_json] - Failed to load table: {e}") from e

    def to_str(self) -> str:
        try:
            if self._table is None:
                raise TableError("No data assigned to table")
            return self._strategy.table_to_str(self._table, self._caption)
        except Exception as e:
            raise TableError(f"[to_str] - Failed to convert table to string: {e}") from e

    def get_caption(self) -> str:
        return self._caption if self._caption else "[No Caption]"


class PandasStrategy(ITableStrategy):
    def operation(
        self, table: pd.DataFrame, operation: str, args: Match[str], table_name: str
    ) -> pd.DataFrame:
        table_logger = get_logger(table_name, "table")
        try:
            match operation:
                case "f_add_column":
                    col_name = args.group(1).strip()
                    values = args.group(2)
                    if " | " in values:
                        values = values.split(" | ")
                    else:
                        values = [values]
                    table[col_name] = values
                case "f_select_column":
                    col_names = args.group(1).strip().split(", ")
                    table = table[col_names]
                case "f_select_row":
                    table_args = args.group(1).strip()
                    if "*" not in table_args:
                        idx = [
                            int(i.split("row ")[1]) - 1
                            for i in table_args.split(",")
                        ]
                        table = table.iloc[idx]
                case "f_sort_by":
                    col_name = args.group(1).strip()
                    ascending = False if args.group(2) == "large to small" else True
                    table = table.sort_values(by=[col_name], ascending=ascending)
                case "f_group_by":
                    col_name = args.group(1).strip()
                    table = table.groupby(by=[col_name]).size().reset_index(name="Count")
                case _:
                    raise RecoverableError(f"Unknown operation name: {operation}")
        except KeyError as e:
            raise RecoverableError from e
        return table

    def get_table_strategy(self) -> str:
        return "Pandas DataFrame"

    def is_compatible_table(self, table: Any) -> bool:
        return isinstance(table, pd.DataFrame)

    def json_to_table(self, json_data) -> Tuple[Any, str]:
        if "table" not in json_data:
            raise ValueError("'table' key not found in JSON data")
        table_data = json_data["table"]
        caption = table_data.get("name", "[No caption]")
        if "header" not in table_data or "rows" not in table_data:
            raise ValueError("Invalid table data")
        return (
            pd.DataFrame(table_data["rows"], columns=table_data["header"]),
            caption,
        )

    def table_to_str(self, table: pd.DataFrame, caption: Optional[str]) -> str:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")
        output = ""
        if caption:
            output += f"table caption : {caption}.\n"
        output += "col : " + " | ".join(table.columns) + "\n"
        for i, (_, row) in enumerate(table.iterrows()):
            row_str = f"row {int(i) + 1} : "
            row_values = [str(item).replace("\n", " ").strip() for item in row.values]
            output += row_str + " | ".join(row_values) + "\n"
        return output