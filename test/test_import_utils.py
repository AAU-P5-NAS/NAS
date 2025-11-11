import pytest

import src.data_module.import_utils as import_utils


def test_transform_google_drive_url_to_direct_download():
    url1 = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
    url2 = "https://drive.google.com/open?id=FILE_ID"
    expected = "https://drive.google.com/uc?export=download&id=FILE_ID"

    assert import_utils.transform_google_drive_url_to_direct_download(url1) == expected
    assert import_utils.transform_google_drive_url_to_direct_download(url2) == expected

    with pytest.raises(ValueError):
        import_utils.transform_google_drive_url_to_direct_download("https://example.com/file")
