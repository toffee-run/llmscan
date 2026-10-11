"""Detectors: independent signals that a file was written by an LLM."""

from .base import Detector, build_detectors, load_detectors, register

__all__ = ["Detector", "build_detectors", "load_detectors", "register"]
