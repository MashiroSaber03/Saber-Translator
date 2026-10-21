//! Catalog metadata, strict tool envelopes and live resources.
use super::*;
use rmcp::RoleServer;
use rmcp::model::*;
use rmcp::service::RequestContext;

const DOCUMENT: &str = "photocraft://document";
const COMMANDS: &str = "photocraft://commands";
const INSTRUCTIONS: &str = "PhotoCraft image editor. Start with doc_new/doc_open, discover ids and parameter docs with command_list, edit with command_run or command_batch (one history step per command). Inspect with doc_inspect and render_preview; doc_save writes native .pcraft or exports by extension. File paths are relative to the read/write roots granted at launch. ui_inspect/ui_screenshot and other ui_* tools require bridge mode. Read photocraft://document and photocraft://commands for live JSON state.";

fn annotated(mut tool: Tool) -> Tool {
    let words = tool.name.replace('_', " ");
    let mut chars = words.chars();
    let title = chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default();
    let read = matches!(
        tool.name.as_ref(),
        "command_list" | "doc_inspect" | "session_list" | "render_preview" | "doc_render_preview" | "ui_inspect" | "ui_screenshot"
    );
    let additive = matches!(tool.name.as_ref(), "doc_new" | "doc_open" | "doc_place");
    tool.title = Some(title.clone());
    tool.annotations = Some(ToolAnnotations::from_raw(Some(title), Some(read), Some(!read && !additive), Some(read), Some(false)));
    Arc::make_mut(&mut tool.input_schema).insert("additionalProperties".into(), json!(false));
    tool
}

impl ServerHandler for PhotocraftMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().build())
            .with_server_info(Implementation::new("photocraft", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(&self, _: Option<PaginatedRequestParams>, _: RequestContext<RoleServer>) -> Result<ListToolsResult, McpError> {
        let tools: Vec<_> = self.tool_router.list_all().into_iter().map(annotated).collect();
        serde_json::from_value(json!({"resultType":"complete", "ttlMs":600000, "cacheScope":"private", "tools":tools}))
            .map_err(|e| McpError::internal_error(e.to_string(), None))
    }

    async fn call_tool(&self, request: CallToolRequestParams, context: RequestContext<RoleServer>) -> Result<CallToolResponse, McpError> {
        if let Some(tool) = self.tool_router.get(&request.name) {
            let properties = tool.input_schema.get("properties").and_then(Value::as_object);
            if let Some(args) = &request.arguments
                && let Some(key) = args.keys().find(|k| properties.is_none_or(|p| !p.contains_key(*k)))
            {
                return Err(McpError::invalid_params(format!("unknown argument `{key}` for {}", request.name), None));
            }
        }
        if request.name == "command_batch" {
            let args = Value::Object(request.arguments.clone().unwrap_or_default());
            let _: BatchParams = serde_json::from_value(args).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        }
        if request.name == "command_run" && matches!(&*self.backend, Backend::Headless(_)) {
            let args = Value::Object(request.arguments.clone().unwrap_or_default());
            let params: RunParams = serde_json::from_value(args).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
            if params.id == "file.export.renderVideo" {
                return Ok(self.render_video_progress(params.params.unwrap_or_else(|| json!({})), context).await.into());
            }
        }
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    async fn list_resources(&self, _: Option<PaginatedRequestParams>, _: RequestContext<RoleServer>) -> Result<ListResourcesResult, McpError> {
        serde_json::from_value(json!({"resultType":"complete", "ttlMs":600000, "cacheScope":"private", "resources":[
            {"uri":DOCUMENT, "name":"document", "title":"Active document", "mimeType":"application/json", "description":"Live doc_inspect state, or an empty session."},
            {"uri":COMMANDS, "name":"commands", "title":"Command catalog", "mimeType":"application/json", "description":"Live command_list including enabled state."}
        ]})).map_err(|e| McpError::internal_error(e.to_string(), None))
    }

    async fn read_resource(&self, request: ReadResourceRequestParams, _: RequestContext<RoleServer>) -> Result<ReadResourceResponse, McpError> {
        let result = match request.uri.as_str() {
            DOCUMENT => self.doc_inspect(Parameters(DocIndex::default())).await?,
            COMMANDS => self.command_list(Parameters(ListParams::default())).await?,
            uri => return Err(McpError::resource_not_found(format!("unknown resource `{uri}`"), None)),
        };
        let text = result.content.iter().filter_map(|c| c.as_text()).map(|t| t.text.as_str()).collect::<Vec<_>>().join("\n");
        if result.is_error == Some(true) {
            return Err(McpError::internal_error(text, None));
        }
        let result: ReadResourceResult = serde_json::from_value(json!({"resultType":"complete", "ttlMs":0, "cacheScope":"private", "contents":[
            {"uri":request.uri, "mimeType":"application/json", "text":text}
        ]}))
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(result.into())
    }
}
