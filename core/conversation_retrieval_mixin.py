from core.conversation_retrieval_backend_mixin import ConversationRetrievalBackendMixin
from core.conversation_retrieval_keyfile_mixin import ConversationRetrievalKeyfileMixin
from core.conversation_retrieval_orchestration_mixin import ConversationRetrievalOrchestrationMixin
from core.conversation_retrieval_postprocess_mixin import ConversationRetrievalPostprocessMixin


class ConversationRetrievalMixin(
    ConversationRetrievalBackendMixin,
    ConversationRetrievalKeyfileMixin,
    ConversationRetrievalPostprocessMixin,
    ConversationRetrievalOrchestrationMixin,
):
    """Facade mixin composing retrieval backend, post-processing, and orchestration concerns."""
