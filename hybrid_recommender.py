
import pickle
import pandas as pd
import numpy as np

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('df_books.pkl', 'rb') as f:
    df_books = pickle.load(f)

with open('final_svdpp_model.pkl', 'rb') as f:
    svdpp_model = pickle.load(f)

df_books['book_id'] = df_books['book_id'].astype(str)
book_indices        = pd.Series(df_books.index, index=df_books['book_id'])

# 2. HÀM CONTENT-BASED
def get_content_scores(book_id, top_n=50):
    book_id = str(book_id)
    if book_id not in book_indices:
        return {}

    idx    = book_indices[book_id]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx][:top_n]

    return {
        str(df_books.iloc[i]['book_id']): round(float(score), 4)
        for i, score in scores
    }

# 3. HÀM COLLABORATIVE FILTERING (SVDpp)
def get_cf_score(user_id, book_id):
    pred = svdpp_model.predict(int(user_id), int(book_id))
    return float(np.clip(pred.est, 1, 5))
    
# 4. HÀM HYBRID CHÍNH — SWITCHING + WEIGHTED
def recommend_hybrid(user_id,
                     rated_book_ids=None,
                     top_n=10,
                     w_cf=0.6,
                     w_cb=0.4,
                     min_ratings_for_cf=3):
    uid = int(user_id) 

    # Lấy sách đã đọc
    if rated_book_ids is None:
        try:
            inner_uid      = svdpp_model.trainset.to_inner_uid(uid)  #  dùng uid 
            rated_book_ids = {
                svdpp_model.trainset.to_raw_iid(iid)
                for iid, _ in svdpp_model.trainset.ur[inner_uid]
            }
            print(f" Tìm thấy {len(rated_book_ids)} ratings trong trainset")
        except ValueError:
            rated_book_ids = set()
            print(f" Không tìm thấy user {user_id} trong trainset")

    rated_book_ids = set(str(b) for b in rated_book_ids)
    is_cold_start  = len(rated_book_ids) < min_ratings_for_cf

    if is_cold_start:
        print(f" User {user_id} là user mới ({len(rated_book_ids)} ratings) "
              f"→ dùng Content-Based (avg_rating)")
    else:
        print(f" User {user_id} có {len(rated_book_ids)} ratings "
              f"→ dùng Weighted Hybrid (CF={w_cf}, CB={w_cb})")

    all_book_ids  = set(df_books['book_id'])
    candidate_ids = all_book_ids - rated_book_ids

    #  Lấy CB scores 
    cb_scores = {}
    if not is_cold_start and rated_book_ids:
        # Warm user: CB dựa trên sách đã đọc
        for read_book in rated_book_ids:
            similar = get_content_scores(read_book, top_n=100)
            for bid, score in similar.items():
                cb_scores[bid] = max(cb_scores.get(bid, 0), score)

    #  Fallback cold-start: dùng avg_rating thay vì 1.0 cho tất cả
    if is_cold_start or not cb_scores:
        print(" Dùng avg_rating làm điểm CB (không có lịch sử đọc)")
        cb_scores = {
            str(row['book_id']): float(row['avg_rating'])
            for _, row in df_books.iterrows()
            if pd.notna(row.get('avg_rating'))
        }

    # Normalize CB scores về [1, 5]
    if cb_scores:
        min_cb   = min(cb_scores.values())
        max_cb   = max(cb_scores.values())
        cb_range = max_cb - min_cb if max_cb != min_cb else 1
        cb_scores = {
            bid: 1 + 4 * (s - min_cb) / cb_range
            for bid, s in cb_scores.items()
        }

    # ── Tính điểm cho từng candidate ─────────────────────────────────────────
    results = []
    for book_id in candidate_ids:
        bid_str  = str(book_id)
        score_cb = cb_scores.get(bid_str, 1.0)

        if is_cold_start:
            score_cf     = None
            score_hybrid = score_cb
        else:
            score_cf     = get_cf_score(user_id, book_id)
            score_hybrid = float(np.clip(
                w_cf * score_cf + w_cb * score_cb, 1, 5
            ))

        results.append({
            'book_id'      : bid_str,
            'score_cf'     : round(score_cf, 4) if score_cf is not None else None,
            'score_cb'     : round(score_cb, 4),
            'score_hybrid' : round(score_hybrid, 4),
        })

    # Sort và lấy top_n
    df_result = (
        pd.DataFrame(results)
        .sort_values('score_hybrid', ascending=False)
        .head(top_n)
        .merge(
            df_books[['book_id','description', 'title', 'author', 'genres', 'avg_rating','image_url']],
            on='book_id', how='left'
        )
        [['book_id', 'title', 'author', 'genres','description','image_url',
          'avg_rating', 'score_cf', 'score_cb', 'score_hybrid']]
        .reset_index(drop=True)
    )
    df_result.index += 1
    return df_result, is_cold_start, len(rated_book_ids)

# 5. ĐÁNH GIÁ HYBRID TRÊN TEST SET
def evaluate_hybrid(testset, w_cf=0.6, w_cb=0.4):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tqdm import tqdm

    y_true, y_cf, y_hybrid = [], [], []

    # Precompute CB scores cho toàn bộ item trong testset
    
    all_iids        = list(set(iid for _, iid, _ in testset))
    cb_scores_cache = {}
    for iid in all_iids:
        similar = get_content_scores(str(iid), top_n=50)
        cb_scores_cache[str(iid)] = max(similar.values()) if similar else 1.0

    # Normalize cache về [1, 5]
    if cb_scores_cache:
        min_cb   = min(cb_scores_cache.values())
        max_cb   = max(cb_scores_cache.values())
        cb_range = max_cb - min_cb if max_cb != min_cb else 1
        cb_scores_cache = {
            k: 1 + 4 * (v - min_cb) / cb_range
            for k, v in cb_scores_cache.items()
        }

    for uid, iid, r_true in tqdm(testset, desc="Evaluating"):
        cf_score     = get_cf_score(uid, iid)
        cb_score     = cb_scores_cache.get(str(iid), 1.0)
        hybrid_score = float(np.clip(w_cf * cf_score + w_cb * cb_score, 1, 5))

        y_true.append(r_true)
        y_cf.append(cf_score)
        y_hybrid.append(hybrid_score)

    rmse_cf     = np.sqrt(mean_squared_error(y_true, y_cf))
    rmse_hybrid = np.sqrt(mean_squared_error(y_true, y_hybrid))
    mae_cf      = mean_absolute_error(y_true, y_cf)
    mae_hybrid  = mean_absolute_error(y_true, y_hybrid)

    df_eval = pd.DataFrame({
        'Model': ['SVDpp (CF only)', 'Hybrid (CF + CB)'],
        'RMSE' : [round(rmse_cf, 4),     round(rmse_hybrid, 4)],
        'MAE'  : [round(mae_cf,  4),     round(mae_hybrid,  4)],
    })
   # display(df_eval)
    return df_eval

# if __name__ == '__main__':

#     # Warm user — có nhiều ratings
#     print("\n" + "="*60)
#     result_warm = recommend_hybrid(user_id=3672777, top_n=10)
#     display(result_warm)

#     # Cold-start user — user mới chưa có ratings
#     print("\n" + "="*60)
#     result_cold = recommend_hybrid(user_id=99999999, top_n=10,
#                                    rated_book_ids=[])
#     display(result_cold)
