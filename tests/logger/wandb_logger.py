import wandb
from typing import Dict, Any, Optional, Union


class WandbLogger:
    """
    A wrapper class for wandb logger that provides authentication, workspace selection,
    and simplified logging with custom tags.
    """

    def __init__(self, name: str, project: str, entity: Optional[str] = None):
        """
        Initialize the WandbLogger.

        Args:
            name: The name of the logger
            project: The name of the project where runs will be logged
            entity: Optional workspace/username where the project is located
        """
        super().__init__(name)
        self.project = project
        self.entity = entity
        self.run = None
        self._is_initialized = False

    def authenticate(self, api_key: Optional[str] = None) -> bool:
        """
        Authenticate to wandb using an API key.

        Args:
            api_key: Optional wandb API key. If not provided, will use the key from wandb's config.
                     You can set it with `wandb login` CLI command.

        Returns:
            bool: True if authentication was successful
        """
        try:
            wandb.login(key=api_key)
            super().log("Successfully authenticated with wandb")
            return True
        except Exception as e:
            super().log(f"Failed to authenticate with wandb: {str(e)}")
            return False

    def set_workspace(self, entity: str) -> None:
        """
        Set the workspace (entity) for wandb logging.

        Args:
            entity: The username or team name to use for logging
        """
        self.entity = entity
        super().log(f"Set wandb workspace to: {entity}")

    def init_run(self, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None) -> None:
        """
        Initialize a new wandb run.

        Args:
            run_name: Optional name for the run
            config: Optional configuration dictionary for the run
            tags: Optional list of tags to associate with the run
        """
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config,
            tags=tags,
            reinit=True
        )
        self._is_initialized = True
        super().log(f"Initialized wandb run: {self.run.name} in project {self.project}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None,
                    commit: bool = True, tags: Optional[list] = None) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            commit: Whether to immediately commit the metrics
            tags: Optional new tags to add to the run
        """
        if not self._is_initialized:
            self.init_run(tags=tags)

        # Add tags if provided
        if tags and self.run:
            current_tags = self.run.tags
            for tag in tags:
                if tag not in current_tags:
                    current_tags.append(tag)
            self.run.tags = current_tags

        wandb.log(metrics, step=step, commit=commit)
        super().log(f"Logged metrics: {list(metrics.keys())}")

    def log_artifact(self, artifact_path: str, name: str, artifact_type: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[list] = None) -> None:
        """
        Log an artifact (file) to wandb.

        Args:
            artifact_path: Path to the artifact file
            name: Name of the artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset')
            metadata: Optional metadata for the artifact
            tags: Optional tags for the artifact
        """
        if not self._is_initialized:
            self.init_run(tags=tags)

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
        artifact.add_file(artifact_path)

        self.run.log_artifact(artifact)
        super().log(f"Logged artifact: {name}")

    def finish(self) -> None:
        """
        Finish the current wandb run.
        """
        if self._is_initialized and self.run:
            self.run.finish()
            self._is_initialized = False
            super().log("Finished wandb run")