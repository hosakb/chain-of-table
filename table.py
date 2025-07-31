import abc
from re import Match
from typing import Any, Optional, Tuple, Union
import logging

import pandas as pd
import sqlite3

from utils import RecoverableError, TableError


class ITableStrategy(abc.ABC):
    def __init__(self):
        self._logger = logging.getLogger("my_logger")

    @abc.abstractmethod
    def operation(self, table: Any, operation: str, args: Match[str]) -> Any:
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
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_current_table_strategy(self) -> str:
        return self._strategy.get_table_strategy()

    def perform_operation(self, operation: str, args: Match[str]):
        try:
            if self._table is None:
                raise TableError("No data assigned to table")

            self._table = self._strategy.operation(self._table, operation, args)
        except RecoverableError:
            raise
        except Exception as e:
            raise TableError(f"[perform_operation] - Failed to perform operation '{operation}': {e}") from e
            


    def load_from_json(self, json_data):
        try:
            self._table, self._caption = self._strategy.json_to_table(json_data)
        except Exception as e:
            raise TableError(f"[load_from_json] - Failed to load table from JSON: {e}") from e
           

    def to_str(self) -> str:
        try:
            if self._table is None:
                raise TableError("No data assigned to table")
            return self._strategy.table_to_str(self._table, self._caption)
        except Exception as e:
            raise TableError(f"[to_str] - Failed to convert table to string: {e}") from e
            

    def get_caption(self) -> str:
        if self._caption is None:
            return "[No Caption]"
        return self._caption


########################################## Strategies #################################################
class PandasStrategy(ITableStrategy):
    def __init__(self):
        super().__init__()

    def operation(
        self, table: pd.DataFrame, operation: str, args: Match[str]
    ) -> pd.DataFrame:

        try:
            match operation:
                case "f_add_column":
                    col_name = args.group(1).strip()
                    values = args.group(2)

                    if " | " in values:
                        values = values.split(" | ")
                    else:
                        values = [values]

                    self._logger.debug(f"[operation] - col_name: {col_name}, args: {values}")

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
                    print(col_name)
                    ascending = False if args.group(2) == "large to small" else True
                    table = table.sort_values(by=[col_name], ascending=ascending)

                case "f_group_by":
                    col_name = args.group(1).strip()
                    table = table.groupby(by=[col_name]).size().reset_index(name="Count")
                case _:
                    raise RecoverableError(f"Unknown operation name provided {operation}")
                
        except KeyError as e:
            raise RecoverableError from e
        except RecoverableError:
            raise
        except Exception:
            raise
        return table

    def get_table_strategy(self) -> str:
        return "Pandas DataFrame"

    def is_compatible_table(self, table: Any) -> bool:
        return isinstance(table, pd.DataFrame)

    def json_to_table(self, json_data) -> Tuple[Any, str]:
        if "table" not in json_data:
            raise ValueError("'table' key not found in JSON data")

        table_data = json_data["table"]

        if "name" not in table_data:
            caption = "[No caption]"
        else:
            caption = table_data["name"]

        if "header" not in table_data:
            raise ValueError("'header' key not found in table data")

        if "rows" not in table_data:
            raise ValueError("'rows' key not found in table data")

        return (
            pd.DataFrame(table_data["rows"], columns=table_data["header"]),
            caption
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


# TODO: impl SQL Strategy
