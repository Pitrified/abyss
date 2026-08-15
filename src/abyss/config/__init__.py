"""Config models: the shape of what a device or a viewer contributes.

Models fix the shape, `abyss.params` supplies the values, pydantic validates.
Every model here describes a **device or a person**, never a machine: the Pixel
records frames another machine processes, so the host says nothing about the
camera the frames came from.
"""
