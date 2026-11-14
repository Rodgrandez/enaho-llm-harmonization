import zipfile

from enahoharm import fetch


def test_module_url():
    assert fetch.module_url(966, "05") == (
        "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/966-Modulo05.zip"
    )


def test_main_dta_picks_base_file(tmp_path):
    zp = tmp_path / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("966-Modulo34/sumaria-2024.dta", b"x" * 100)
        z.writestr("966-Modulo34/sumaria-2024-12g.dta", b"x" * 500)
        z.writestr("966-Modulo34/Sumaria-2024.pdf", b"x" * 900)
    assert fetch.main_dta(zp, 2024, "34") == "966-Modulo34/sumaria-2024.dta"


def test_download_skips_existing(tmp_path):
    target = tmp_path / "2024" / "m34.dta"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")

    def opener(*args, **kwargs):
        raise AssertionError("network must not be called")

    assert fetch.download(2024, "34", tmp_path, opener=opener) == target


def test_download_extracts_main_member(tmp_path):
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("966-Modulo34/sumaria-2024.dta", b"payload")
        z.writestr("966-Modulo34/Sumaria-2024.pdf", b"pdf")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    out = fetch.download(2024, "34", tmp_path, opener=lambda *a, **k: Resp(buf.getvalue()))
    assert out == tmp_path / "2024" / "m34.dta"
    assert out.read_bytes() == b"payload"
    assert not list((tmp_path / "2024").glob("*.zip"))
