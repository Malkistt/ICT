from pydantic import BaseModel

class MustahikInput(BaseModel):
    penghasilan: float
    jumlah_tanggungan: int
    pekerjaan: str
    kondisi_rumah: str
    lokasi: str