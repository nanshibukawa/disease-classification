"""
Componente para divisão manual dos dados em train/validation/test
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from cnnClassifier import logger


class DataSplitter:
    def __init__(self, source_dir: Path, output_dir: Path, train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, test_ratio: float = 0.1, random_state: int = 42):
        """
        Inicializa o divisor de dados
        
        Args:
            source_dir: Diretório fonte com estrutura classe/imagens
            output_dir: Diretório de saída para train/val/test
            train_ratio: Proporção para treino (padrão: 70%)
            val_ratio: Proporção para validação (padrão: 20%) 
            test_ratio: Proporção para teste (padrão: 10%)
            random_state: Seed para reprodutibilidade
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Validar proporções
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Soma das proporções deve ser 1.0, obtido: {total_ratio}")
        
        logger.info(f"DataSplitter inicializado: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    def get_class_files(self) -> dict:
        """
        Obtém lista de arquivos por classe
        
        Returns:
            dict: {classe: [lista_de_arquivos]}
        """
        class_files = {}
        
        for class_dir in self.source_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                files = [f for f in class_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                class_files[class_name] = files
                logger.info(f"Classe '{class_name}': {len(files)} arquivos encontrados")
        
        return class_files
    
    def split_files_by_class(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Divide arquivos de uma classe em train/val/test
        
        Args:
            files: Lista de arquivos da classe
            
        Returns:
            tuple: (train_files, val_files, test_files)
        """
        # Primeiro split: separar test
        train_val_files, test_files = train_test_split(
            files, 
            test_size=self.test_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Segundo split: separar train e validation do restante
        # Ajustar proporções para o split restante
        remaining_ratio = self.train_ratio + self.val_ratio
        val_ratio_adjusted = self.val_ratio / remaining_ratio
        
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            shuffle=True
        )
        
        return train_files, val_files, test_files
    
    def create_split_directories(self):
        """Cria estrutura de diretórios para train/val/test"""
        
        class_files = self.get_class_files()
        
        for split in ['train', 'validation', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for class_name in class_files.keys():
                class_split_dir = split_dir / class_name
                class_split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Estrutura de diretórios criada em: {self.output_dir}")
    
    def copy_files_to_split(self, files: List[Path], destination_dir: Path, class_name: str):
        """
        Copia arquivos para o diretório de destino
        
        Args:
            files: Lista de arquivos para copiar
            destination_dir: Diretório de destino (train/val/test)
            class_name: Nome da classe
        """
        class_dest_dir = destination_dir / class_name
        
        for file_path in files:
            dest_file = class_dest_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_file)
            except Exception as e:
                logger.error(f"Erro ao copiar {file_path} para {dest_file}: {e}")
    
    def perform_split(self) -> dict:
        """
        Executa a divisão completa dos dados
        
        Returns:
            dict: Estatísticas da divisão
        """
        logger.info("Iniciando divisão dos dados...")
        
        # Limpar diretório de saída se existir
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"Diretório existente removido: {self.output_dir}")
        
        # Criar estrutura de diretórios
        self.create_split_directories()
        
        # Obter arquivos por classe
        class_files = self.get_class_files()
        
        # Estatísticas finais
        stats = {
            'train': {},
            'validation': {},
            'test': {},
            'total': {}
        }
        
        # Processar cada classe
        for class_name, files in class_files.items():
            logger.info(f"Processando classe: {class_name}")
            
            # Dividir arquivos
            train_files, val_files, test_files = self.split_files_by_class(files)
            
            # Copiar arquivos para diretórios correspondentes
            self.copy_files_to_split(train_files, self.output_dir / 'train', class_name)
            self.copy_files_to_split(val_files, self.output_dir / 'validation', class_name)
            self.copy_files_to_split(test_files, self.output_dir / 'test', class_name)
            
            # Atualizar estatísticas
            stats['train'][class_name] = len(train_files)
            stats['validation'][class_name] = len(val_files)
            stats['test'][class_name] = len(test_files)
            stats['total'][class_name] = len(files)
            
            logger.info(f"  Classe '{class_name}': Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Calcular totais
        for split in ['train', 'validation', 'test', 'total']:
            stats[f'{split}_total'] = sum(stats[split].values())
        
        logger.info(f"Divisão concluída:")
        logger.info(f"  Train: {stats['train_total']} arquivos ({stats['train_total']/stats['total_total']:.1%})")
        logger.info(f"  Validation: {stats['validation_total']} arquivos ({stats['validation_total']/stats['total_total']:.1%})")
        logger.info(f"  Test: {stats['test_total']} arquivos ({stats['test_total']/stats['total_total']:.1%})")
        logger.info(f"  Total: {stats['total_total']} arquivos")
        
        # Salvar estatísticas
        self.save_split_stats(stats)
        
        return stats
    
    def save_split_stats(self, stats: dict):
        """Salva estatísticas da divisão em arquivo JSON"""
        from cnnClassifier.utils.common import save_json
        
        stats_file = self.output_dir / 'split_statistics.json'
        save_json(stats_file, stats)
        logger.info(f"Estatísticas salvas em: {stats_file}")
    
    def verify_split(self) -> bool:
        """
        Verifica se a divisão foi realizada corretamente
        
        Returns:
            bool: True se a divisão está correta
        """
        try:
            splits = ['train', 'validation', 'test']
            
            for split in splits:
                split_dir = self.output_dir / split
                if not split_dir.exists():
                    logger.error(f"Diretório {split} não encontrado")
                    return False
                
                # Verificar se há arquivos em cada classe
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                if not class_dirs:
                    logger.error(f"Nenhuma classe encontrada em {split}")
                    return False
                
                for class_dir in class_dirs:
                    files = [f for f in class_dir.iterdir() if f.is_file()]
                    if not files:
                        logger.warning(f"Nenhum arquivo encontrado em {class_dir}")
            
            logger.info("Verificação da divisão: OK")
            return True
            
        except Exception as e:
            logger.error(f"Erro na verificação: {e}")
            return False


class DataSplitterConfig:
    """Configuração para o DataSplitter"""
    
    def __init__(
        self,
        source_dir: str = "artifacts/data_ingestion/bycbh73438-1",
        output_dir: str = "artifacts/data_split",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_state: int = 42
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state


def create_data_split():
    """Função utilitária para criar divisão dos dados"""
    config = DataSplitterConfig()
    splitter = DataSplitter(
        source_dir=config.source_dir,
        output_dir=config.output_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state
    )
    
    stats = splitter.perform_split()
    
    if splitter.verify_split():
        logger.info("Divisão dos dados realizada com sucesso!")
        return stats
    else:
        logger.error("Falha na divisão dos dados!")
        return None


if __name__ == "__main__":
    # Executar divisão diretamente
    create_data_split()
