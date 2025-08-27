import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RagDocument:
  """ Represent a document with its content and metadata """
  id: str
  content: str
  metadata: Dict[str,Any]=None
  embedding: Optional[np.ndarray]=None