//! Render Video runs on the existing blocking worker; rmcp remains available for ping/cancel.
use super::*;
use rmcp::RoleServer;
use rmcp::model::ProgressNotificationParam;
use rmcp::service::RequestContext;
use std::time::{Duration, Instant};

impl PhotocraftMcp {
    pub(super) async fn render_video_progress(&self, params: Value, context: RequestContext<RoleServer>) -> CallToolResult {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // rmcp may drop the request future on cancellation. The blocking worker owns the
        // token too, so it still stops and cleans up without needing that future to be polled.
        let cancel = context.ct.clone();
        let job = self.headless_op(move |h| {
            h.render_video(params, move |done, total| {
                let _ = tx.send((done, total));
                !cancel.is_cancelled()
            })
        });
        tokio::pin!(job);
        let token = context.meta.get_progress_token();
        let (mut last, mut sent_at): (Option<usize>, Option<Instant>) = (None, None);
        let result = loop {
            tokio::select! {
                biased;
                Some((done, total)) = rx.recv() => {
                    let due = done == total || sent_at.is_none_or(|t| t.elapsed() >= Duration::from_millis(100));
                    if let Some(token) = token.clone().filter(|_| due && last.is_none_or(|n| done > n) && !context.ct.is_cancelled()) {
                        last = Some(done);
                        sent_at = Some(Instant::now());
                        let _ = context.peer.notify_progress(ProgressNotificationParam::new(token, done as f64).with_total(total as f64)).await;
                    }
                }
                result = &mut job => break result,
            }
        };
        match result {
            Some(Ok(value)) => ok_json(&value),
            Some(Err(error)) => fail(error),
            None => no_backend(),
        }
    }
}
