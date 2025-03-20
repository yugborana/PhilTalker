from generate_embeddings import vespa_feed
from vespa.deployment import VespaCloud
from vespa import colsmol_schema
from vespa.package import ApplicationPackage
app_name = "philtalker"
tenant_name = "yugborana"
vespa_application_package = ApplicationPackage(name=app_name, schema=[colsmol_schema])

vespa_cloud = VespaCloud(tenant=tenant_name, application=app_name, application_package=vespa_application_package)

app: Vespa = vespa_cloud.deploy()