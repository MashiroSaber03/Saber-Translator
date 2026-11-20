"""Model downloads are server-wide administration, separate from user settings."""

from flask import Blueprint, jsonify

from src.backend_v2.api.request_helpers import error_response, json_body
from src.backend_v2.auth.context import require_admin
from src.backend_v2.model_installer import ModelInstallBusy, ModelInstaller
from src.backend_v2.runtime_profile import RuntimeProfile


def create_model_blueprint(*, installer: ModelInstaller, profile: RuntimeProfile) -> Blueprint:
    blueprint = Blueprint("models_v2", __name__, url_prefix="/api/v2/system/models")

    @blueprint.errorhandler(ValueError)
    def validation(error):
        return error_response("validation_error", str(error), 422)

    @blueprint.errorhandler(ModelInstallBusy)
    def busy(error):
        return error_response("model_install_busy", str(error), 409)

    @blueprint.errorhandler(OSError)
    def storage_error(error):
        return error_response("model_storage_error", str(error), 422)

    @blueprint.errorhandler(PermissionError)
    def forbidden(error):
        return error_response("forbidden", str(error) or "需要管理员权限", 403)

    def require_management():
        if profile.requires_auth:
            require_admin()

    @blueprint.get("")
    def list_models():
        # Paths and progress are server administration details in public mode.
        require_management()
        return jsonify(installer.snapshot())

    @blueprint.post("/install")
    def install_models():
        require_management()
        body = json_body(allowed_keys={"modelIds", "source", "verify"})
        verify = body.get("verify", False)
        if not isinstance(verify, bool):
            raise ValueError("verify 必须是布尔值")
        source = body.get("source", "auto")
        if not isinstance(source, str):
            raise ValueError("source 必须是字符串")
        return jsonify(installer.install(body.get("modelIds"), source=source, verify=verify)), 202

    @blueprint.post("/prepare-valen")
    def prepare_valen():
        require_management()
        json_body(allowed_keys=(), optional=True)
        return jsonify(installer.prepare_valen()), 202

    @blueprint.post("/cancel")
    def cancel_install():
        require_management()
        json_body(allowed_keys=(), optional=True)
        return jsonify(installer.cancel()), 202

    return blueprint
