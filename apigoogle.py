from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import gdown
import json
import os
import hashlib
from enum import Enum

MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
MAX_FILES = int(os.environ.get("MAX_FILES", "5"))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# ===== MODELLI PYDANTIC =====

class AggregationType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"

class TableInfo(BaseModel):
    name: str
    rows: int
    columns: int
    date_column: Optional[str] = None

class ColumnInfo(BaseModel):
    name: str
    type: str

class TableSchema(BaseModel):
    table: str
    date_column: Optional[str]
    columns: List[ColumnInfo]

class FilterCondition(BaseModel):
    """Condizione di filtro per una colonna"""
    column: str = Field(..., description="Nome della colonna")
    operator: str = Field(..., description="Operatore: eq, ne, gt, lt, gte, lte, in, contains")
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(..., description="Valore da confrontare")
    
    class Config:
        schema_extra = {
            "example": {
                "column": "city",
                "operator": "eq",
                "value": "Rome"
            }
        }

# 🆕 NUOVO: Modello per caricare CSV
class LoadCSVRequest(BaseModel):
    """Richiesta per caricare un nuovo CSV"""
    table_name: str = Field(..., description="Nome da assegnare alla tabella", example="my_data")
    gdrive_url: str = Field(..., description="URL pubblico di Google Drive", example="https://drive.google.com/file/d/ABC123/view")
    date_column: Optional[str] = Field(None, description="Nome della colonna con le date (opzionale)", example="date")
    overwrite: bool = Field(False, description="Sovrascrivi se la tabella esiste già")
    
    class Config:
        schema_extra = {
            "example": {
                "table_name": "pollution_data",
                "gdrive_url": "https://drive.google.com/file/d/1BJrfprXh1picyTQusVO6oW5zRxWQ_up7/view?usp=sharing",
                "date_column": "reftime",
                "overwrite": False
            }
        }

class QueryRequest(BaseModel):
    """Richiesta di query con filtri avanzati"""
    columns: Optional[List[str]] = Field(None, description="Colonne da selezionare")
    filters: Optional[List[FilterCondition]] = Field(None, description="Condizioni di filtro")
    limit: int = Field(100, ge=1, le=10000, description="Numero massimo di risultati")
    offset: int = Field(0, ge=0, description="Offset per paginazione")
    
    # 🆕 OPZIONALE: Carica CSV al volo
    gdrive_url: Optional[str] = Field(None, description="URL Google Drive (carica al volo se non in cache)")
    date_column_hint: Optional[str] = Field(None, description="Colonna date per CSV caricato al volo")
    
    class Config:
        schema_extra = {
            "example": {
                "columns": ["date", "temperature", "city"],
                "filters": [
                    {"column": "city", "operator": "eq", "value": "Rome"},
                    {"column": "temperature", "operator": "gt", "value": 20}
                ],
                "limit": 100,
                "offset": 0
            }
        }

class AggregationRequest(BaseModel):
    """Richiesta di aggregazione con filtri"""
    column: str = Field(..., description="Colonna numerica da aggregare")
    agg_type: AggregationType = Field(AggregationType.MEAN, description="Tipo di aggregazione")
    filters: Optional[List[FilterCondition]] = Field(None, description="Filtri da applicare prima dell'aggregazione")
    days: Optional[int] = Field(None, ge=1, le=365, description="Finestra mobile in giorni")
    yearly_average: bool = Field(False, description="Calcola media annuale per giorno")
    
    # 🆕 OPZIONALE: Carica CSV al volo
    gdrive_url: Optional[str] = Field(None, description="URL Google Drive (carica al volo se non in cache)")
    date_column_hint: Optional[str] = Field(None, description="Colonna date per CSV caricato al volo")
    
    class Config:
        schema_extra = {
            "example": {
                "column": "temperature",
                "agg_type": "mean",
                "filters": [
                    {"column": "city", "operator": "eq", "value": "Rome"}
                ],
                "days": 30,
                "yearly_average": False
            }
        }

# ===== DATA MANAGER =====

class DataManager:
    """Gestisce il caricamento e la gestione dei CSV"""
    
    def __init__(self):
        self.tables: Dict[str, pd.DataFrame] = {}
        self.schemas: Dict[str, Dict[str, str]] = {}
        self.date_columns: Dict[str, Optional[str]] = {}
        # 🆕 Cache per tracciare l'origine dei dati
        self.table_sources: Dict[str, str] = {}  # table_name -> gdrive_url
        self.url_cache: Dict[str, str] = {}  # url_hash -> table_name
        
    def _hash_url(self, url: str) -> str:
        """Genera hash univoco per URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def get_or_load_from_url(self, gdrive_url: str, date_column: Optional[str] = None) -> str:
        """
        🆕 Carica un CSV da URL o restituisce il nome della tabella se già in cache
        
        Returns:
            Nome della tabella caricata
        """
        url_hash = self._hash_url(gdrive_url)
        
        # Controlla se già in cache
        if url_hash in self.url_cache:
            table_name = self.url_cache[url_hash]
            print(f"♻️ Using cached table '{table_name}' for URL")
            return table_name
        
        # Carica nuovo CSV
        table_name = f"temp_{url_hash}"
        self.add_csv_from_gdrive(table_name, gdrive_url, date_column)
        self.url_cache[url_hash] = table_name
        
        return table_name
        
    def add_csv_from_gdrive(self, table_name: str, gdrive_url: str, date_column: Optional[str] = None):
        """
        Scarica e carica un CSV da Google Drive
        
        Args:
            table_name: Nome della tabella
            gdrive_url: URL pubblico di Google Drive
            date_column: Nome della colonna che contiene le date (opzionale)
        """
        try:
            # Estrai l'ID del file dall'URL
            if '/file/d/' in gdrive_url:
                file_id = gdrive_url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in gdrive_url:
                file_id = gdrive_url.split('id=')[1].split('&')[0]
            else:
                file_id = gdrive_url
            
            # Scarica il file
            download_url = f'https://drive.google.com/uc?id={file_id}'
            output_path = f'temp_{table_name}.csv'
            
            print(f"📥 Downloading {table_name} from Google Drive...")
            gdown.download(download_url, output_path, quiet=False)
            
            # Carica il CSV
            self.add_csv_from_file(table_name, output_path, date_column)
            
            # 🆕 Salva l'origine
            self.table_sources[table_name] = gdrive_url
            
            # Rimuovi il file temporaneo
            Path(output_path).unlink(missing_ok=True)
            
        except Exception as e:
            raise Exception(f"Error loading {table_name}: {str(e)}")
    
    def add_csv_from_file(self, table_name: str, file_path: str, date_column: Optional[str] = None):
        """Carica un CSV da file locale"""
        try:
            # Carica il CSV con inferenza automatica dei tipi
            df = pd.read_csv(file_path, low_memory=False)
            
            # Prova a convertire le colonne date
            if date_column and date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                self.date_columns[table_name] = date_column
            else:
                # Cerca automaticamente colonne con date
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_datetime(df[col], errors='raise')
                            if date_column is None:
                                self.date_columns[table_name] = col
                                date_column = col
                            break
                        except:
                            continue
            
            self.tables[table_name] = df
            self._generate_schema(table_name)
            print(f"✅ Loaded {table_name}: {len(df):,} rows, {len(df.columns)} columns")
            
        except Exception as e:
            raise Exception(f"Error loading {table_name}: {str(e)}")
    
    def remove_table(self, table_name: str) -> bool:
        """🆕 Rimuove una tabella dalla memoria"""
        if table_name not in self.tables:
            return False
        
        # Rimuovi da tutte le strutture dati
        del self.tables[table_name]
        del self.schemas[table_name]
        
        if table_name in self.date_columns:
            del self.date_columns[table_name]
        
        if table_name in self.table_sources:
            # Rimuovi anche dalla URL cache
            url = self.table_sources[table_name]
            url_hash = self._hash_url(url)
            if url_hash in self.url_cache:
                del self.url_cache[url_hash]
            del self.table_sources[table_name]
        
        print(f"🗑️ Removed table '{table_name}'")
        return True
    
    def _generate_schema(self, table_name: str):
        """Genera lo schema della tabella con i tipi di dato"""
        df = self.tables[table_name]
        schema = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                schema[col] = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema[col] = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                schema[col] = "boolean"
            else:
                schema[col] = "string"
        
        self.schemas[table_name] = schema
    
    def _apply_filters(self, df: pd.DataFrame, filters: List[FilterCondition]) -> pd.DataFrame:
        """Applica una lista di filtri a un DataFrame"""
        if not filters:
            return df
        
        df_filtered = df.copy()
        
        for filter_cond in filters:
            col = filter_cond.column
            op = filter_cond.operator
            val = filter_cond.value
            
            if col not in df_filtered.columns:
                raise ValueError(f"Column '{col}' not found")
            
            # Applica operatore
            if op == "eq":
                df_filtered = df_filtered[df_filtered[col] == val]
            elif op == "ne":
                df_filtered = df_filtered[df_filtered[col] != val]
            elif op == "gt":
                df_filtered = df_filtered[df_filtered[col] > val]
            elif op == "lt":
                df_filtered = df_filtered[df_filtered[col] < val]
            elif op == "gte":
                df_filtered = df_filtered[df_filtered[col] >= val]
            elif op == "lte":
                df_filtered = df_filtered[df_filtered[col] <= val]
            elif op == "in":
                if not isinstance(val, list):
                    raise ValueError(f"Operator 'in' requires a list value")
                df_filtered = df_filtered[df_filtered[col].isin(val)]
            elif op == "contains":
                df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(str(val), na=False)]
            else:
                raise ValueError(f"Unknown operator: {op}")
        
        return df_filtered
    
    def get_table_list(self) -> List[Dict[str, Any]]:
        """Restituisce l'elenco delle tabelle con informazioni base"""
        return [
            {
                "name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "date_column": self.date_columns.get(name),
                "source": self.table_sources.get(name, "unknown")  # 🆕
            }
            for name, df in self.tables.items()
        ]
    
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Restituisce lo schema di una tabella"""
        if table_name not in self.schemas:
            raise ValueError(f"Table '{table_name}' not found")
        
        return {
            "table": table_name,
            "date_column": self.date_columns.get(table_name),
            "columns": [
                {"name": col, "type": dtype}
                for col, dtype in self.schemas[table_name].items()
            ]
        }
    
    def query_table(self, table_name: str, query_request: QueryRequest) -> Dict[str, Any]:
        """Interroga una tabella con filtri avanzati"""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.tables[table_name].copy()
        
        # Applica filtri
        if query_request.filters:
            df = self._apply_filters(df, query_request.filters)
        
        # Seleziona colonne
        if query_request.columns:
            missing_cols = set(query_request.columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df = df[query_request.columns]
        
        # Paginazione
        total = len(df)
        df = df.iloc[query_request.offset:query_request.offset + query_request.limit]
        
        # Converti in dict gestendo i tipi speciali
        result = self._dataframe_to_dict(df)
        
        return {
            "total": total,
            "offset": query_request.offset,
            "limit": query_request.limit,
            "returned": len(result),
            "data": result
        }
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converte un DataFrame in lista di dizionari gestendo tipi speciali"""
        result = df.replace({np.nan: None}).to_dict(orient='records')
        
        # Converti tipi speciali in tipi JSON-serializzabili
        for record in result:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    record[key] = value.isoformat()
                elif isinstance(value, np.integer):
                    record[key] = int(value)
                elif isinstance(value, np.floating):
                    record[key] = float(value)
        
        return result
    
    def aggregate_column(
        self,
        table_name: str,
        agg_request: AggregationRequest
    ) -> Dict[str, Any]:
        """
        Calcola aggregazioni su una colonna numerica con filtri opzionali
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.tables[table_name].copy()
        
        # Applica filtri PRIMA dell'aggregazione
        if agg_request.filters:
            df = self._apply_filters(df, agg_request.filters)
            if len(df) == 0:
                raise ValueError("No data remaining after applying filters")
        
        column = agg_request.column
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        schema_type = self.schemas[table_name].get(column)
        if schema_type not in ["integer", "float"]:
            raise ValueError(f"Column '{column}' is not numeric (type: {schema_type})")
        
        date_col = self.date_columns.get(table_name)
        
        if agg_request.yearly_average:
            return self._calculate_yearly_average(df, column, date_col, agg_request.agg_type, agg_request.filters)
        elif agg_request.days:
            return self._calculate_rolling_aggregate(df, column, date_col, agg_request.agg_type, agg_request.days, agg_request.filters)
        else:
            return self._calculate_simple_aggregate(df, column, agg_request.agg_type, agg_request.filters)
    
    def _calculate_simple_aggregate(
        self, 
        df: pd.DataFrame, 
        column: str, 
        agg_type: AggregationType,
        filters: Optional[List[FilterCondition]] = None
    ) -> Dict[str, Any]:
        """Calcola aggregazione semplice su tutta la colonna"""
        data = df[column].dropna()
        
        if agg_type == AggregationType.MEAN:
            value = float(data.mean())
        elif agg_type == AggregationType.MEDIAN:
            value = float(data.median())
        elif agg_type == AggregationType.MAX:
            value = float(data.max())
        elif agg_type == AggregationType.MIN:
            value = float(data.min())
        
        return {
            "column": column,
            "aggregation": agg_type.value,
            "filters_applied": len(filters) if filters else 0,
            "value": value,
            "count": len(data),
            "std_dev": float(data.std()) if len(data) > 1 else None
        }
    
    def _calculate_rolling_aggregate(
        self,
        df: pd.DataFrame,
        column: str,
        date_col: Optional[str],
        agg_type: AggregationType,
        days: int,
        filters: Optional[List[FilterCondition]] = None
    ) -> Dict[str, Any]:
        """Calcola aggregazione mobile su N giorni"""
        if not date_col or date_col not in df.columns:
            raise ValueError("Date column required for rolling aggregation")
        
        # Ordina per data
        df_sorted = df[[date_col, column]].copy().sort_values(date_col)
        df_sorted = df_sorted.dropna(subset=[column])
        
        # Crea rolling window
        df_sorted.set_index(date_col, inplace=True)
        
        if agg_type == AggregationType.MEAN:
            result = df_sorted[column].rolling(window=f'{days}D').mean()
        elif agg_type == AggregationType.MEDIAN:
            result = df_sorted[column].rolling(window=f'{days}D').median()
        elif agg_type == AggregationType.MAX:
            result = df_sorted[column].rolling(window=f'{days}D').max()
        elif agg_type == AggregationType.MIN:
            result = df_sorted[column].rolling(window=f'{days}D').min()
        
        # Converti in lista di dict
        result_data = [
            {
                "date": idx.isoformat(),
                "value": float(val) if not pd.isna(val) else None
            }
            for idx, val in result.items()
        ]
        
        return {
            "column": column,
            "aggregation": agg_type.value,
            "filters_applied": len(filters) if filters else 0,
            "days": days,
            "data_points": len(result_data),
            "data": result_data
        }
    
    def _calculate_yearly_average(
        self,
        df: pd.DataFrame,
        column: str,
        date_col: Optional[str],
        agg_type: AggregationType,
        filters: Optional[List[FilterCondition]] = None
    ) -> Dict[str, Any]:
        """Calcola la media annuale per ogni giorno dell'anno"""
        if not date_col or date_col not in df.columns:
            raise ValueError("Date column required for yearly average")
        
        df_copy = df[[date_col, column]].copy().dropna(subset=[column])
        
        # Estrai giorno dell'anno (1-366)
        df_copy['day_of_year'] = df_copy[date_col].dt.dayofyear
        df_copy['month'] = df_copy[date_col].dt.month
        df_copy['day'] = df_copy[date_col].dt.day
        
        # Raggruppa per giorno dell'anno e calcola aggregazione
        if agg_type == AggregationType.MEAN:
            grouped = df_copy.groupby('day_of_year')[column].mean()
        elif agg_type == AggregationType.MEDIAN:
            grouped = df_copy.groupby('day_of_year')[column].median()
        elif agg_type == AggregationType.MAX:
            grouped = df_copy.groupby('day_of_year')[column].max()
        elif agg_type == AggregationType.MIN:
            grouped = df_copy.groupby('day_of_year')[column].min()
        
        # Ottieni mese e giorno rappresentativi
        day_info = df_copy.groupby('day_of_year')[['month', 'day']].first()
        
        result_data = [
            {
                "day_of_year": int(day_num),
                "month": int(day_info.loc[day_num, 'month']),
                "day": int(day_info.loc[day_num, 'day']),
                "value": float(val)
            }
            for day_num, val in grouped.items()
        ]
        
        return {
            "column": column,
            "aggregation": f"{agg_type.value}_yearly",
            "filters_applied": len(filters) if filters else 0,
            "description": f"Average {agg_type.value} for each day across all years",
            "data_points": len(result_data),
            "data": result_data
        }

# ===== INIZIALIZZAZIONE FASTAPI =====

app = FastAPI(
    title="CSV Data API - Dynamic Loader",
    description="""
    🚀 **API per interrogare qualsiasi CSV da Google Drive dinamicamente**
    
    ## 🆕 Nuove funzionalità
    
    * 📤 **Caricamento dinamico** - Carica CSV al volo via API
    * 🗑️ **Gestione tabelle** - Aggiungi e rimuovi tabelle in runtime
    * ♻️ **Cache intelligente** - Evita download ripetuti dello stesso file
    * 🔗 **Query diretta da URL** - Interroga CSV senza pre-caricamento
    
    ## Funzionalità principali
    
    * 📊 **Caricamento automatico** di CSV da Google Drive
    * 🔍 **Query avanzate** con filtri multipli
    * 📈 **Aggregazioni statistiche** (media, mediana, max, min)
    * 📅 **Analisi temporali** (medie mobili, medie annuali)
    * 🎯 **Filtri pre-aggregazione** per analisi mirate
    
    ## 🎯 Come usare
    
    ### 1️⃣ Carica un CSV:
    ```bash
    POST /tables/load
    {
        "table_name": "my_data",
        "gdrive_url": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "date_column": "date"
    }
    ```
    
    ### 2️⃣ Interroga i dati:
    ```bash
    POST /tables/my_data/query
    {
        "columns": ["date", "temperature"],
        "limit": 100
    }
    ```
    
    ### 3️⃣ Oppure interroga direttamente da URL:
    ```bash
    POST /query/direct
    {
        "gdrive_url": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "columns": ["date", "temperature"],
        "limit": 100
    }
    ```
    
    ## Operatori di filtro disponibili
    
    * `eq`: uguale a
    * `ne`: diverso da
    * `gt`: maggiore di
    * `lt`: minore di
    * `gte`: maggiore o uguale a
    * `lte`: minore o uguale a
    * `in`: contenuto in una lista
    * `contains`: contiene (per stringhe)
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Abilita CORS per permettere chiamate da qualsiasi origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Istanza globale del data manager
data_manager = DataManager()

# ===== ENDPOINTS API =====

@app.get("/", 
    tags=["Info"],
    summary="Informazioni API",
    description="Restituisce informazioni generali sull'API e gli endpoint disponibili"
)
def root():
    """Informazioni sull'API"""
    return {
        "name": "CSV Data API - Dynamic Loader",
        "version": "3.0.0",
        "description": "API for querying any CSV from Google Drive dynamically",
        "documentation": "/docs",
        "new_features": {
            "load_csv": "POST /tables/load - Load CSV from Google Drive",
            "remove_table": "DELETE /tables/{table_name} - Remove table from memory",
            "query_direct": "POST /query/direct - Query CSV directly from URL"
        },
        "endpoints": {
            "tables": {
                "url": "/tables",
                "description": "List all loaded tables"
            },
            "load": {
                "url": "POST /tables/load",
                "description": "Load new CSV from Google Drive"
            },
            "schema": {
                "url": "/tables/{table_name}/schema",
                "description": "Get table schema"
            },
            "query": {
                "url": "/tables/{table_name}/query",
                "description": "Query table with filters"
            },
            "aggregate": {
                "url": "/tables/{table_name}/aggregate",
                "description": "Aggregate numeric columns with filters"
            }
        }
    }

# 🆕 NUOVO ENDPOINT: Carica CSV dinamicamente
@app.post("/tables/load",
    tags=["Tables"],
    summary="🆕 Carica CSV da Google Drive",
    description="""
    Carica un nuovo CSV da Google Drive nella memoria dell'API.
    
    **Come ottenere l'URL di Google Drive:**
    1. Carica il file su Google Drive
    2. Fai click destro → "Get link" / "Ottieni link"
    3. Assicurati che sia impostato su "Anyone with the link" / "Chiunque abbia il link"
    4. Copia l'URL (es: `https://drive.google.com/file/d/ABC123/view`)
    
    **Esempio:**
    ```json
    {
        "table_name": "pollution_data",
        "gdrive_url": "https://drive.google.com/file/d/1BJrfprXh1picyTQusVO6oW5zRxWQ_up7/view",
        "date_column": "reftime",
        "overwrite": false
    }
    ```
    """,
    response_description="Conferma del caricamento con informazioni sulla tabella"
)
def load_csv_from_gdrive(request: LoadCSVRequest):
    """Carica un nuovo CSV da Google Drive"""
    try:
        # Controlla se la tabella esiste già
        if request.table_name in data_manager.tables and not request.overwrite:
            raise HTTPException(
                status_code=409, 
                detail=f"Table '{request.table_name}' already exists. Set overwrite=true to replace it."
            )
        
        # Carica il CSV
        data_manager.add_csv_from_gdrive(
            table_name=request.table_name,
            gdrive_url=request.gdrive_url,
            date_column=request.date_column
        )
        
        # Restituisci informazioni sulla tabella caricata
        df = data_manager.tables[request.table_name]
        
        return {
            "status": "success",
            "message": f"Table '{request.table_name}' loaded successfully",
            "table_info": {
                "name": request.table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "date_column": data_manager.date_columns.get(request.table_name),
                "source": request.gdrive_url
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

# 🆕 NUOVO ENDPOINT: Rimuovi tabella
@app.delete("/tables/{table_name}",
    tags=["Tables"],
    summary="🆕 Rimuovi tabella",
    description="Rimuove una tabella dalla memoria dell'API"
)
def remove_table(table_name: str):
    """Rimuove una tabella dalla memoria"""
    if data_manager.remove_table(table_name):
        return {
            "status": "success",
            "message": f"Table '{table_name}' removed successfully"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

# 🆕 NUOVO ENDPOINT: Query diretta da URL
@app.post("/query/direct",
    tags=["Query"],
    summary="🆕 Query diretta da URL",
    description="""
    Interroga un CSV direttamente da Google Drive senza doverlo caricare prima.
    
    Il file viene scaricato e messo in cache automaticamente. Le richieste successive
    sullo stesso URL useranno la versione cached.
    
    **Esempio:**
    ```json
    {
        "gdrive_url": "https://drive.google.com/file/d/1BJrfprXh1picyTQusVO6oW5zRxWQ_up7/view",
        "date_column_hint": "reftime",
        "columns": ["reftime", "lat", "lon"],
        "filters": [
            {"column": "lat", "operator": "gt", "value": 40}
        ],
        "limit": 100
    }
    ```
    """
)
def query_direct_from_url(query_request: QueryRequest):
    """Interroga un CSV direttamente da URL (con cache)"""
    if not query_request.gdrive_url:
        raise HTTPException(status_code=400, detail="gdrive_url is required for direct query")
    
    try:
        # Ottieni o carica la tabella
        table_name = data_manager.get_or_load_from_url(
            gdrive_url=query_request.gdrive_url,
            date_column=query_request.date_column_hint
        )
        
        # Esegui la query
        result = data_manager.query_table(table_name, query_request)
        result["cached_table"] = table_name
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 🆕 NUOVO ENDPOINT: Aggregazione diretta da URL
@app.post("/aggregate/direct",
    tags=["Aggregation"],
    summary="🆕 Aggregazione diretta da URL",
    description="Calcola aggregazioni su un CSV direttamente da Google Drive"
)
def aggregate_direct_from_url(agg_request: AggregationRequest):
    """Aggrega dati direttamente da URL (con cache)"""
    if not agg_request.gdrive_url:
        raise HTTPException(status_code=400, detail="gdrive_url is required for direct aggregation")
    
    try:
        # Ottieni o carica la tabella
        table_name = data_manager.get_or_load_from_url(
            gdrive_url=agg_request.gdrive_url,
            date_column=agg_request.date_column_hint
        )
        
        # Esegui l'aggregazione
        result = data_manager.aggregate_column(table_name, agg_request)
        result["cached_table"] = table_name
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tables",
    tags=["Tables"],
    summary="Lista tutte le tabelle",
    description="Restituisce l'elenco di tutte le tabelle caricate in memoria",
    response_model=Dict[str, List[TableInfo]]
)
def list_tables():
    """Elenca tutte le tabelle disponibili"""
    return {
        "tables": data_manager.get_table_list(),
        "count": len(data_manager.tables)
    }

@app.get("/tables/{table_name}/schema",
    tags=["Tables"],
    summary="Schema di una tabella",
    description="Restituisce lo schema completo di una tabella: nomi delle colonne e tipi di dato",
    response_model=TableSchema
)
def get_table_schema(table_name: str):
    """Ottiene lo schema di una tabella specifica"""
    try:
        return data_manager.get_schema(table_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/tables/{table_name}/query",
    tags=["Query"],
    summary="Interroga una tabella",
    description="""
    Interroga una tabella con filtri avanzati e paginazione.
    
    **Esempio di richiesta:**
    ```json
    {
        "columns": ["date", "temperature", "city"],
        "filters": [
            {"column": "city", "operator": "eq", "value": "Rome"},
            {"column": "temperature", "operator": "gt", "value": 20}
        ],
        "limit": 100,
        "offset": 0
    }
    ```
    """,
    response_description="Risultati della query con metadati di paginazione"
)
def query_table_post(
    table_name: str,
    query_request: QueryRequest = Body(
        ...,
        example={
            "columns": ["date", "temperature", "city"],
            "filters": [
                {"column": "city", "operator": "eq", "value": "Rome"},
                {"column": "temperature", "operator": "gt", "value": 20}
            ],
            "limit": 50,
            "offset": 0
        }
    )
):
    """Interroga una tabella (metodo POST con body JSON)"""
    try:
        result = data_manager.query_table(table_name, query_request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tables/{table_name}/query",
    tags=["Query"],
    summary="Interroga una tabella (GET)",
    description="Versione GET semplificata per query veloci senza filtri complessi"
)
def query_table_get(
    table_name: str,
    columns: Optional[str] = Query(None, description="Colonne separate da virgola (es: col1,col2,col3)", example="date,temperature,city"),
    limit: int = Query(100, ge=1, le=10000, description="Numero massimo di risultati"),
    offset: int = Query(0, ge=0, description="Offset per paginazione")
):
    """Interroga una tabella (metodo GET semplificato)"""
    try:
        cols = columns.split(',') if columns else None
        query_request = QueryRequest(columns=cols, limit=limit, offset=offset)
        result = data_manager.query_table(table_name, query_request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tables/{table_name}/aggregate",
    tags=["Aggregation"],
    summary="Aggrega una colonna numerica",
    description="""
    Calcola aggregazioni statistiche su una colonna numerica con possibilità di:
    - Applicare **filtri** prima dell'aggregazione
    - Calcolare **medie mobili** su N giorni (1-365)
    - Calcolare **medie annuali** per ogni giorno dell'anno
    
    **Esempi:**
    
    1️⃣ **Media semplice con filtro:**
    ```json
    {
        "column": "temperature",
        "agg_type": "mean",
        "filters": [
            {"column": "city", "operator": "eq", "value": "Rome"}
        ]
    }
    ```
    
    2️⃣ **Media mobile 30 giorni filtrata per città:**
    ```json
    {
        "column": "temperature",
        "agg_type": "mean",
        "days": 30,
        "filters": [
            {"column": "city", "operator": "in", "value": ["Rome", "Milan"]}
        ]
    }
    ```
    
    3️⃣ **Media annuale per ogni giorno (es. temperatura media del 15 gennaio):**
    ```json
    {
        "column": "temperature",
        "agg_type": "mean",
        "yearly_average": true,
        "filters": [
            {"column": "year", "operator": "gte", "value": 2010}
        ]
    }
    ```
    """,
    response_description="Risultato dell'aggregazione con metadati"
)
def aggregate_column_post(
    table_name: str,
    agg_request: AggregationRequest = Body(
        ...,
        examples={
            "simple_mean": {
                "summary": "Media semplice",
                "description": "Calcola la media di una colonna",
                "value": {
                    "column": "temperature",
                    "agg_type": "mean"
                }
            },
            "filtered_mean": {
                "summary": "Media con filtro",
                "description": "Media filtrata per città",
                "value": {
                    "column": "temperature",
                    "agg_type": "mean",
                    "filters": [
                        {"column": "city", "operator": "eq", "value": "Rome"}
                    ]
                }
            },
            "rolling_mean": {
                "summary": "Media mobile 30 giorni",
                "description": "Media mobile su 30 giorni",
                "value": {
                    "column": "temperature",
                    "agg_type": "mean",
                    "days": 30,
                    "filters": [
                        {"column": "city", "operator": "eq", "value": "Rome"}
                    ]
                }
            },
            "yearly_average": {
                "summary": "Media annuale per giorno",
                "description": "Media per ogni giorno dell'anno su più anni",
                "value": {
                    "column": "temperature",
                    "agg_type": "mean",
                    "yearly_average": True
                }
            }
        }
    )
):
    """Calcola aggregazioni su una colonna numerica"""
    try:
        result = data_manager.aggregate_column(table_name, agg_request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tables/{table_name}/aggregate",
    tags=["Aggregation"],
    summary="Aggrega una colonna (GET semplice)",
    description="Versione GET semplificata per aggregazioni veloci senza filtri",
    deprecated=True
)
def aggregate_column_get(
    table_name: str,
    column: str = Query(..., description="Nome della colonna da aggregare", example="temperature"),
    agg_type: AggregationType = Query(AggregationType.MEAN, description="Tipo di aggregazione"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Finestra mobile in giorni"),
    yearly_average: bool = Query(False, description="Calcola media annuale per giorno")
):
    """Calcola aggregazioni (metodo GET semplificato - preferire POST)"""
    try:
        agg_request = AggregationRequest(
            column=column,
            agg_type=agg_type,
            days=days,
            yearly_average=yearly_average
        )
        result = data_manager.aggregate_column(table_name, agg_request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===== CONFIGURAZIONE INIZIALE (OPZIONALE) =====

def load_default_datasets():
    """
    🆕 OPZIONALE: Carica dataset di default all'avvio
    
    Puoi commentare questa funzione se vuoi caricare SOLO dataset dinamici via API
    """
    
    # Esempio: carica un dataset di default
    # data_manager.add_csv_from_gdrive(
    #     table_name="pollini",
    #     gdrive_url="https://drive.google.com/file/d/1BJrfprXh1picyTQusVO6oW5zRxWQ_up7/view?usp=sharing",
    #     date_column="reftime"
    # )
    
    print("✅ Default datasets loaded (if any)")

@app.on_event("startup")
async def startup_event():
    """Eseguito all'avvio del server"""
    print("\n" + "="*60)
    print("🚀 Starting CSV Data API Server - Dynamic Loader v3.0")
    print("="*60)
    load_default_datasets()  # Commenta se non vuoi dataset di default
    print("="*60)
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📖 Alternative docs: http://localhost:8000/redoc")
    print("💡 Use POST /tables/load to load CSV dynamically")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





