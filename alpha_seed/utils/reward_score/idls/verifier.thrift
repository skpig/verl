include "base.thrift"
namespace py seed.alphaseed.verifier

# batch
typedef list<string> StringList
typedef list<i64> IntList


struct VerifyRequest {
    1: required string reference_answer;
    2: required string generated_response;
    3: required string problem;
    4: required i32 verify_type = 0; # 0 for truncation last 100; 1 for boxed_verify; 2 for final_answer_verify; 3 for <Answer></Answer>
    255: optional base.Base base_p;
}

struct VerifyResponse {
    1: required bool is_correct;
    255: optional base.BaseResp base_resp;
}


service VerifyService {
    VerifyResponse verify(1: VerifyRequest req),
}

