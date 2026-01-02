import React, {useState,useEffect} from 'react'


interface ModalProps {
  conversationId: number | null; // Pass this from the table row click
}


interface FullConversationData {
  user_prompt: string | null;
  system_prompt: string | null;
  agent_response: string | null;
  testcase_name: string | null;
  conversation_id?: number | null;
  target: string | null;
}

function Modal({ conversationId }: ModalProps) {
  const [data, setData] = useState<FullConversationData>({
    user_prompt: null,
    system_prompt: null,
    agent_response: null,
    testcase_name: null,
    conversation_id: null,
    target: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!conversationId) return;
    const fetchData = async () => {
        setLoading(true);
        setError(null);

        try {
        const res = await fetch(`http://localhost:8000/conversations/full/${conversationId}`);
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const json: FullConversationData = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [conversationId]);

  return (
     <div
        className="modal fade"
        id="conversationModal"
        tabIndex={-1}
        aria-labelledby="conversationModalLabel"
        aria-hidden="true"
        >
        <div className="modal-dialog modal-xl modal-dialog-scrollable">
            <div className="modal-content">
            
            <div className="modal-header">
                <h5 className="modal-title" id="conversationModalLabel">
                Conversation Details
                </h5>
                <button
                type="button"
                className="btn-close"
                data-bs-dismiss="modal"
                aria-label="Close"
                />
            </div>

            <div className="modal-body">
                <p><strong>Conversation ID:</strong> {data.conversation_id || "No conversation ID"}</p>
                <p><strong>Testcase:</strong> {data.testcase_name || "No testcase name"}</p>
                <p><strong>Target:</strong> {data.target || "No target"}</p>

                <hr />

                 <h6>User Prompt</h6>
                <pre className="bg-light p-3 rounded text-wrap" style={{ whiteSpace: "pre-wrap" }}>{data.user_prompt || "No user prompt"}</pre>

                <h6 className="mt-3">System Prompt</h6>
                <pre className="bg-light p-3 rounded text-wrap" style={{ whiteSpace: "pre-wrap" }}>{data.system_prompt || "No system prompt"}</pre>

                <h6 className="mt-3">Agent Response</h6>
                <pre className="bg-light p-3 rounded text-wrap" style={{ whiteSpace: "pre-wrap" }}>
                    {data.agent_response || "No response available"}
                </pre>

                <h6 className="mt-3">Evaluation</h6>
                <p><strong>Score:</strong> 0.82</p>
                <p><strong>Reason:</strong> Response follows safety guidelines.</p>
            </div>

            <div className="modal-footer">
                <button
                type="button"
                className="btn btn-secondary"
                data-bs-dismiss="modal"
                >
                Close
                </button>
            </div>

            </div>
        </div>
        </div>
  )
}

export default Modal