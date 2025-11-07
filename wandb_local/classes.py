"""
WandB compatible classes for structured data and media
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np


class Table:
    """Table for structured data logging"""
    
    def __init__(self, columns: List[str] = None, data: List[List[Any]] = None, 
                 dataframe=None, **kwargs):
        self.columns = columns or []
        self.data = data or []
        self.kwargs = kwargs
        
        if dataframe is not None:
            # Convert pandas DataFrame
            self.columns = list(dataframe.columns)
            self.data = dataframe.values.tolist()
            
    def add_column(self, name: str, data: List[Any]):
        """Add a column to the table"""
        if name not in self.columns:
            self.columns.append(name)
            # Fill existing rows with None
            for i in range(len(self.data)):
                if len(self.data[i]) < len(self.columns):
                    self.data[i].append(None)
                    
        col_idx = self.columns.index(name)
        for i, value in enumerate(data):
            if i >= len(self.data):
                self.data.append([None] * len(self.columns))
            self.data[i][col_idx] = value
            
    def add_data(self, *args, **kwargs):
        """Add a row to the table"""
        row = [None] * len(self.columns)
        
        # Handle positional args
        for i, value in enumerate(args):
            if i < len(self.columns):
                row[i] = value
                
        # Handle keyword args
        for key, value in kwargs.items():
            if key in self.columns:
                col_idx = self.columns.index(key)
                row[col_idx] = value
                
        self.data.append(row)
        
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation"""
        return {
            "_type": "table",
            "columns": self.columns,
            "data": self.data,
            "ncols": len(self.columns),
            "nrows": len(self.data)
        }
        
    def save(self, base_dir: str):
        """Save table to file"""
        table_path = os.path.join(base_dir, "media", f"table_{int(time.time() * 1000)}.json")
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        with open(table_path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)


class Media:
    """Base class for media objects"""
    
    def __init__(self, data_or_path: Union[str, np.ndarray], **kwargs):
        self.data_or_path = data_or_path
        self.kwargs = kwargs
        self._caption = kwargs.get('caption', '')
        
    @property
    def caption(self):
        return self._caption
        
    @caption.setter
    def caption(self, value):
        self._caption = value
        
    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError
        
    def save(self, base_dir: str):
        raise NotImplementedError


class Image(Media):
    """Image media object"""
    
    def __init__(self, data_or_path: Union[str, np.ndarray], **kwargs):
        super().__init__(data_or_path, **kwargs)
        self.mode = kwargs.get('mode', 'RGB')
        
    def to_json(self) -> Dict[str, Any]:
        if isinstance(self.data_or_path, str):
            return {
                "_type": "image-file",
                "path": self.data_or_path,
                "caption": self.caption
            }
        else:
            # Convert numpy array to base64
            import PIL.Image as PILImage
            import io
            
            if isinstance(self.data_or_path, np.ndarray):
                # Handle different array shapes
                if len(self.data_or_path.shape) == 2:
                    # Grayscale
                    img = PILImage.fromarray(self.data_or_path.astype(np.uint8), 'L')
                elif len(self.data_or_path.shape) == 3:
                    if self.data_or_path.shape[2] == 1:
                        img = PILImage.fromarray(self.data_or_path[:,:,0].astype(np.uint8), 'L')
                    elif self.data_or_path.shape[2] == 3:
                        img = PILImage.fromarray(self.data_or_path.astype(np.uint8), 'RGB')
                    elif self.data_or_path.shape[2] == 4:
                        img = PILImage.fromarray(self.data_or_path.astype(np.uint8), 'RGBA')
                    else:
                        raise ValueError(f"Unsupported image shape: {self.data_or_path.shape}")
                else:
                    raise ValueError(f"Unsupported image shape: {self.data_or_path.shape}")
            else:
                raise ValueError("Unsupported image data type")
                
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "_type": "image-base64",
                "data": img_str,
                "format": "png",
                "caption": self.caption
            }
            
    def save(self, base_dir: str):
        """Save image to file"""
        if isinstance(self.data_or_path, str):
            # Copy existing file
            import shutil
            src_path = self.data_or_path
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(base_dir, "media", filename)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
        else:
            # Save numpy array as image
            import PIL.Image as PILImage
            
            timestamp = int(time.time() * 1000)
            filename = f"image_{timestamp}.png"
            image_path = os.path.join(base_dir, "media", filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            if isinstance(self.data_or_path, np.ndarray):
                if len(self.data_or_path.shape) == 2:
                    img = PILImage.fromarray(self.data_or_path.astype(np.uint8), 'L')
                elif len(self.data_or_path.shape) == 3:
                    img = PILImage.fromarray(self.data_or_path.astype(np.uint8), 'RGB')
                else:
                    raise ValueError(f"Unsupported image shape: {self.data_or_path.shape}")
                    
            img.save(image_path)


class Audio(Media):
    """Audio media object"""
    
    def __init__(self, data_or_path: Union[str, np.ndarray], sample_rate: int = 44100, **kwargs):
        super().__init__(data_or_path, **kwargs)
        self.sample_rate = sample_rate
        
    def to_json(self) -> Dict[str, Any]:
        if isinstance(self.data_or_path, str):
            return {
                "_type": "audio-file",
                "path": self.data_or_path,
                "sample_rate": self.sample_rate,
                "caption": self.caption
            }
        else:
            return {
                "_type": "audio-numpy",
                "sample_rate": self.sample_rate,
                "shape": self.data_or_path.shape if hasattr(self.data_or_path, 'shape') else None,
                "caption": self.caption
            }
            
    def save(self, base_dir: str):
        """Save audio to file"""
        if isinstance(self.data_or_path, str):
            # Copy existing file
            import shutil
            src_path = self.data_or_path
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(base_dir, "media", filename)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
        else:
            # Save numpy array as audio (WAV format)
            import scipy.io.wavfile as wavfile
            
            timestamp = int(time.time() * 1000)
            filename = f"audio_{timestamp}.wav"
            audio_path = os.path.join(base_dir, "media", filename)
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Ensure data is in correct format
            if isinstance(self.data_or_path, np.ndarray):
                # Normalize to [-1, 1] if needed
                audio_data = self.data_or_path.astype(np.float32)
                if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
                # Convert to 16-bit PCM
                audio_data = (audio_data * 32767).astype(np.int16)
                
            wavfile.write(audio_path, self.sample_rate, audio_data)


class Video(Media):
    """Video media object"""
    
    def __init__(self, data_or_path: Union[str, np.ndarray], fps: int = 4, **kwargs):
        super().__init__(data_or_path, **kwargs)
        self.fps = fps
        self.format = kwargs.get('format', 'mp4')
        
    def to_json(self) -> Dict[str, Any]:
        if isinstance(self.data_or_path, str):
            return {
                "_type": "video-file",
                "path": self.data_or_path,
                "fps": self.fps,
                "format": self.format,
                "caption": self.caption
            }
        else:
            return {
                "_type": "video-numpy",
                "fps": self.fps,
                "format": self.format,
                "shape": self.data_or_path.shape if hasattr(self.data_or_path, 'shape') else None,
                "caption": self.caption
            }
            
    def save(self, base_dir: str):
        """Save video to file"""
        if isinstance(self.data_or_path, str):
            # Copy existing file
            import shutil
            src_path = self.data_or_path
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dest_path = os.path.join(base_dir, "media", filename)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
        else:
            # Save numpy array as video
            timestamp = int(time.time() * 1000)
            filename = f"video_{timestamp}.{self.format}"
            video_path = os.path.join(base_dir, "media", filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # For simplicity, save as sequence of images
            # In a full implementation, you'd use ffmpeg or similar
            if isinstance(self.data_or_path, np.ndarray) and len(self.data_or_path.shape) == 4:
                # Shape: (frames, height, width, channels)
                frames_dir = os.path.join(base_dir, "media", f"video_frames_{timestamp}")
                os.makedirs(frames_dir, exist_ok=True)
                
                import PIL.Image as PILImage
                for i, frame in enumerate(self.data_or_path):
                    if len(frame.shape) == 3:
                        img = PILImage.fromarray(frame.astype(np.uint8), 'RGB')
                    else:
                        img = PILImage.fromarray(frame.astype(np.uint8), 'L')
                    img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))


class Artifact:
    """Artifact for data version management"""
    
    def __init__(self, name: str, type: str = None, description: str = None,
                 metadata: Dict[str, Any] = None, **kwargs):
        self.name = name
        self.type = type or "dataset"
        self.description = description or ""
        self.metadata = metadata or {}
        self.kwargs = kwargs
        
        # Internal state
        self.files = {}
        self.refs = {}
        self.digest = None
        self.version = "latest"
        self._finalized = False
        
    def add_file(self, local_path: str, name: str = None):
        """Add a file to the artifact"""
        if self._finalized:
            raise RuntimeError("Cannot add files to finalized artifact")
            
        if name is None:
            name = os.path.basename(local_path)
            
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
            
        self.files[name] = {
            "local_path": local_path,
            "size": os.path.getsize(local_path),
            "mtime": os.path.getmtime(local_path)
        }
        
    def add_dir(self, local_path: str, name: str = None):
        """Add a directory to the artifact"""
        if self._finalized:
            raise RuntimeError("Cannot add files to finalized artifact")
            
        if name is None:
            name = os.path.basename(local_path.rstrip('/'))
            
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Directory not found: {local_path}")
            
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_path)
                artifact_name = os.path.join(name, rel_path)
                self.add_file(file_path, artifact_name)
                
    def add_reference(self, uri: str, name: str = None):
        """Add a reference to external data"""
        if name is None:
            name = os.path.basename(uri)
            
        self.refs[name] = {
            "uri": uri,
            "type": "external"
        }
        
    def get_path(self, name: str):
        """Get path to artifact file"""
        if name in self.files:
            return self.files[name]["local_path"]
        elif name in self.refs:
            return self.refs[name]["uri"]
        else:
            raise FileNotFoundError(f"File not found in artifact: {name}")
            
    def finalize(self):
        """Finalize the artifact"""
        if self._finalized:
            return
            
        # Compute digest
        import hashlib
        hasher = hashlib.md5()
        
        for name in sorted(self.files.keys()):
            file_info = self.files[name]
            hasher.update(name.encode())
            hasher.update(str(file_info["size"]).encode())
            hasher.update(str(file_info["mtime"]).encode())
            
        self.digest = hasher.hexdigest()
        self._finalized = True
        
    def save(self, base_dir: str):
        """Save artifact metadata"""
        self.finalize()
        
        artifact_dir = os.path.join(base_dir, "artifacts", self.name)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Copy files to artifact directory
        for name, file_info in self.files.items():
            dest_path = os.path.join(artifact_dir, name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            import shutil
            shutil.copy2(file_info["local_path"], dest_path)
            
        # Save artifact metadata
        metadata = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "metadata": self.metadata,
            "files": {name: {
                "size": info["size"],
                "mtime": info["mtime"]
            } for name, info in self.files.items()},
            "refs": self.refs,
            "digest": self.digest,
            "version": self.version,
            "created_at": time.time()
        }
        
        metadata_path = os.path.join(artifact_dir, "artifact_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation"""
        return {
            "_type": "artifact",
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "digest": self.digest,
            "version": self.version,
            "num_files": len(self.files),
            "num_refs": len(self.refs)
        }