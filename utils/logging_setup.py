"""
logging_setup.py — Configura logging consistente para todo el pipeline
"""
import logging
import sys
from pathlib import Path

def setup_logging(name="pipeline", output_dir=None):
    """
    Configura logger con salida a archivo y consola.
    
    Args:
        name: nombre del logger
        output_dir: directorio para log file (default: output/)
    
    Returns:
        logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Evita duplicar handlers
    if logger.handlers:
        return logger
    
    # Formato
    fmt = logging.Formatter(
        '[%(levelname)-8s | %(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    
    # Handler archivo (opcional)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            output_dir / "pipeline.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name="pipeline"):
    """Obtiene logger configurado."""
    return logging.getLogger(name)
