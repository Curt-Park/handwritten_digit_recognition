import pytest
import tempfile
import os
import shutil
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file in the temp directory."""
    temp_file_path = os.path.join(temp_dir, 'test_file.txt')
    with open(temp_file_path, 'w') as f:
        f.write('test content')
    return temp_file_path


@pytest.fixture
def sample_mnist_data():
    """Generate sample MNIST-like data for testing."""
    # Create sample training data
    x_train = np.random.rand(1000, 28, 28, 1).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000)
    
    # Create sample test data
    x_test = np.random.rand(200, 28, 28, 1).astype(np.float32)
    y_test = np.random.randint(0, 10, 200)
    
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture
def sample_model_predictions():
    """Generate sample model predictions for ensemble testing."""
    num_samples = 100
    num_classes = 10
    
    predictions = {
        'mobilenet': np.random.rand(num_samples, num_classes),
        'vgg16': np.random.rand(num_samples, num_classes),
        'resnet164': np.random.rand(num_samples, num_classes),
        'wide_resnet': np.random.rand(num_samples, num_classes)
    }
    
    # Normalize to make them look like probabilities
    for key in predictions:
        predictions[key] = predictions[key] / predictions[key].sum(axis=1, keepdims=True)
    
    return predictions


@pytest.fixture
def mock_keras_model():
    """Create a mock Keras model for testing."""
    model = MagicMock()
    model.compile = MagicMock()
    model.fit = MagicMock()
    model.evaluate = MagicMock(return_value=[0.1, 0.95])  # [loss, accuracy]
    model.predict = MagicMock(return_value=np.random.rand(100, 10))
    model.save = MagicMock()
    model.load_weights = MagicMock()
    model.summary = MagicMock()
    return model


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        'epochs': 10,
        'batch_size': 64,
        'data_augmentation': True,
        'save_model_and_weights': True,
        'load_weights': False,
        'plot_training_progress': False,
        'save_model_to_image': False,
        'path_for_weights': './models/test_model.h5',
        'path_for_image': './images/test_model.png',
        'path_for_plot': './images/test_model_plot.png'
    }


@pytest.fixture
def models_dir(temp_dir):
    """Create a temporary models directory."""
    models_path = os.path.join(temp_dir, 'models')
    os.makedirs(models_path, exist_ok=True)
    return models_path


@pytest.fixture
def images_dir(temp_dir):
    """Create a temporary images directory."""
    images_path = os.path.join(temp_dir, 'images')
    os.makedirs(images_path, exist_ok=True)
    return images_path


@pytest.fixture
def predictions_dir(temp_dir):
    """Create a temporary predictions directory."""
    predictions_path = os.path.join(temp_dir, 'predictions')
    os.makedirs(predictions_path, exist_ok=True)
    return predictions_path


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Store original directory
    original_cwd = os.getcwd()
    yield
    # Restore original directory after test
    os.chdir(original_cwd)


@pytest.fixture
def numpy_random_seed():
    """Set numpy random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset random state after test
    np.random.seed(None)


@pytest.fixture
def sample_ensemble_weights():
    """Generate sample ensemble weights for testing."""
    return {
        'mobilenet': 0.25,
        'vgg16': 0.25,
        'resnet164': 0.25,
        'wide_resnet': 0.25
    }