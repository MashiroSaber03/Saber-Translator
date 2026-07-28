class Plugin:
    def before_job(self, context, data):
        context.logger.info(
            "job started",
            jobId=context.job_id,
            mode=context.mode,
        )
        return dict(data)

    def after_job(self, context, data):
        context.logger.info(
            "job finished",
            jobId=context.job_id,
            mode=context.mode,
        )
        return dict(data)

    def before_pipeline(self, context, data):
        if context.config.get("log_target", True):
            context.logger.info(
                "pipeline started",
                bookId=context.book_id,
                chapterId=context.chapter_id,
            )
        return dict(data)

    def after_pipeline(self, context, data):
        context.logger.info(
            "pipeline finished",
            jobId=context.job_id,
        )
        return dict(data)
