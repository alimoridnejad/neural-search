from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage

from utils import logger


def get_vespa_app(app_name, schema):
    """
    This method builds and deploys the application package and returns the app.
    Returns:
    """
    logger.info(
        f"building {app_name} application using schema: {schema.name}, ann deploying using Vespa docker"
    )

    # build the application package
    app_package = ApplicationPackage(name=app_name, schema=[schema])

    # deploy the application
    vespa_docker = VespaDocker()
    app = vespa_docker.deploy(application_package=app_package)

    return app
