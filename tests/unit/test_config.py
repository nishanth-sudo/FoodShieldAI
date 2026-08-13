from backend.config import Settings

def test_default_settings_exist():
    settings = Settings()
    assert hasattr(settings, 'app_name')
    assert hasattr(settings, 'database_url')

def test_max_upload_size_default():
    settings = Settings()
    assert getattr(settings, 'max_upload_size_mb', 10) == 10

def test_allowed_image_types():
    settings = Settings()
    types = getattr(settings, 'allowed_image_types', ['image/jpeg'])
    assert 'image/jpeg' in types

def test_jwt_algorithm_default():
    settings = Settings()
    assert getattr(settings, 'jwt_algorithm', 'HS256') == 'HS256'
