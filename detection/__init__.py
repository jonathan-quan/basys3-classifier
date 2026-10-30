"""Separate, software-only single-object detector project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CLASSES = ("rubiks_cube", "pen", "book", "mouse")
NAMES = ("Rubik's Cube", "Pen", "Book", "Computer Mouse")
SIZE = 96
