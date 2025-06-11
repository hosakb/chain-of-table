import abc
from re import Match
from typing import Any, Optional, Tuple, Union

import pandas as pd
import sqlite3

class ITableStrategy(abc.ABC):
    @abc.abstractmethod
    def operation(self, table: Any, operation: str, args: Match[str]) -> Any: # none in the case of using a SQL DB
        pass
    @abc.abstractmethod
    def get_table_strategy(self) -> str:
        pass
    @abc.abstractmethod
    def json_to_table(self, json_data) -> Tuple[Any, str]:
        pass
    @abc.abstractmethod
    def table_to_str(self, table: Any, table_caption="") -> str:
        pass
    @abc.abstractmethod
    def is_compatible_table(self, table: Any) -> bool:
        pass
    
    
class Table:
    _strategy: ITableStrategy

    def __init__(self, strategy: ITableStrategy):
        if not isinstance(strategy, ITableStrategy):
            raise TypeError("Provided strategy must be an instance of ITableStrategy.")
        self._strategy = strategy
        self._table: Union[pd.DataFrame, sqlite3.Connection, None] = None
        self._caption: Optional[str]

    def get_current_table_strategy(self) -> str:
        return self._strategy.get_table_strategy()
    
    def perform_operation(self, operation: str, args: Match[str]):
        if self._table is None:
            raise ValueError("No table data available")
        
        self._table = self._strategy.operation(self._table, operation, args)
    
    def load_from_json(self, json_data):
        self._table, self._caption = self._strategy.json_to_table(json_data)
    
    def to_str(self) -> str:
        if self._table is None:
            raise ValueError("[to_str] - No table data available")
        if self._caption is None:
            raise ValueError("[to_str] - No table caption available")
        return self._strategy.table_to_str(self._table, self._caption)
    
    def get_caption(self) -> str:
        if self._caption is None:
            raise ValueError(f"No caption for table available for table")
        return self._caption
    

########################################## Strategies #################################################
class PandasStrategy(ITableStrategy):
    def operation(self, table: pd.DataFrame, operation: str, args: Match[str]) -> pd.DataFrame:

        if table is None:
            raise ValueError("No DataFrame assigned to table")

        match operation:
            case "f_add_column":
                col_name = args.group(1).strip()
                values = args.group(2).split(" | ")
                table[col_name] = values
            case "f_select_column":
                col_name = args.group(1).strip()
                table = table[[col_name]]
            case "f_select_row":
                    idx = args.group(1).strip().split(" ")[1]
                    table = table.iloc[[int(idx) - 1]]
            case "f_sort_by":
                col = args.group(1).strip()
                ascending = False if args.group(2) == "large to small" else True
                table = table.sort_values(by=[col], ascending=ascending)
            case "f_group_by":
                col = args.group(1).strip()
                table = table.groupby(by=[col]).size().reset_index(name='Count')
            case _:
                raise NotImplementedError()
            
        return table
    

    def get_table_strategy(self) -> str:
        return "Pandas DataFrame"
    
    def is_compatible_table(self, table: Any) -> bool:
        return isinstance(table, pd.DataFrame)
    
    def json_to_table(self, json_data) -> Tuple[Any, str]:
        if 'table' not in json_data:
            raise ValueError("'table' key not found in JSON data")

        table_data = json_data['table']

        if 'name' not in table_data:
            raise ValueError("'caption' key not found in table data")

        if 'header' not in table_data:
            raise ValueError("'header' key not found in table data")

        if 'rows' not in table_data:
            raise ValueError("'rows' key not found in table data")

        return pd.DataFrame(table_data['rows'], columns=table_data['header']), table_data['name']
    
    def table_to_str(self, table: pd.DataFrame, caption: str) -> str:

        if not isinstance(table, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        output = f"table caption : {caption}.\n"

        output += "col : " + " | ".join(table.columns) + "\n"

        for i, (_, row) in enumerate(table.iterrows()):
            row_str = f"row {int(i) + 1} : "
            row_values = [str(item).replace('\n', ' ').strip() for item in row.values]
            output += row_str + " | ".join(row_values) + "\n"

        return output
    
# TODO impl SQL Strategy
    
