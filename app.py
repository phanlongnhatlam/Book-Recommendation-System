import streamlit as st
import numpy as np
import pandas as pd
from hybrid_recommender import (
    recommend_hybrid,
    get_content_scores,
    cosine_sim,
    df_books,
    svdpp_model,
    book_indices,
)
import gspread
from google.oauth2.service_account import Credentials

def save_rating_to_sheets(new_rows):
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # 1. Lấy dữ liệu từ Streamlit Secrets và chuyển thành Dictionary chuẩn
        gcp_info = dict(st.secrets["gcp_service_account"])
        
        # 2. FIX LỖI CHÍNH Ở ĐÂY: Xử lý ký tự xuống dòng bị lỗi
        if "private_key" in gcp_info:
            gcp_info["private_key"] = gcp_info["private_key"].replace("\\n", "\n")
            
        # 3. Sử dụng bộ key đã được "dọn dẹp" sạch sẽ
        creds  = Credentials.from_service_account_info(
            gcp_info, scopes=scope
        )
        client = gspread.authorize(creds)
        sheet_url = "https://docs.google.com/spreadsheets/d/1ZWdWhZS_esFTYXCqmmUodqdMdWWbGM3yG-LeA8E1zv0/edit"
        sheet  = client.open_by_url(sheet_url).sheet1

        # Chuyển DataFrame thành list of lists để append một lần
        rows_to_append = []
        for _, row in new_rows.iterrows():
            rows_to_append.append([
                str(row['user_id']),
                str(row['book_id']),
                str(row['rating']),
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            
        # Dùng append_rows (có chữ s)
        sheet.append_rows(rows_to_append)
        return True
        
    except Exception as e:
        # Tạm thời vẫn giữ raise e để nếu có lỗi khác nó sẽ báo thẳng ra web cho mình dễ fix
        raise e

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    top_n = st.slider("Số sách gợi ý", 5, 20, 10)
    w_cf  = st.slider("Trọng số CF (SVDpp)", 0.0, 1.0, 0.6, 0.05)
    w_cb  = round(1 - w_cf, 2)
    st.info(f"Trọng số CB: **{w_cb}**")
    st.divider()
    st.markdown(f"Tổng số sách: `{len(df_books):,}`")
    st.markdown("Model: SVDpp + TF-IDF Cosine")
    st.markdown("Hybrid: Switching + Weighted")


# ── Hàm helper ───────────────────────────────────────────────────────────────
def recommend_by_title(book_title, top_n=10):
    title_to_id = pd.Series(df_books['book_id'].values, index=df_books['title'])
    if book_title not in title_to_id.index:
        return None
    book_id = str(title_to_id[book_title])
    similar = get_content_scores(book_id, top_n=top_n + 1)
    if not similar:
        return None
    top_ids = list(similar.keys())[:top_n]
    scores  = list(similar.values())[:top_n]
    return (
        pd.DataFrame({'book_id': top_ids, 'similarity': scores})
        .merge(df_books, on='book_id', how='left')
        .reset_index(drop=True)
    )


def render_book_card(row, score_label=None, score_value=None):
    col_img, col_info = st.columns([1, 3])
    with col_img:
        # Lấy link ảnh. LƯU Ý: Nếu file CSV của bạn tên cột ảnh là khác (VD: 'Image-URL-M'), hãy sửa chữ 'image_url' ở dưới.
        img_url = row.get('image_url') 
        
        # Kiểm tra xem link có hợp lệ không và bọc Try-Except để chống sập web do link chết
        if pd.notna(img_url) and isinstance(img_url, str) and img_url.startswith('http'):
            try:
                st.image(img_url, width=120)
            except:
                st.markdown("<h1>📖</h1>", unsafe_allow_html=True) # Nếu ảnh chết thì hiện icon
        else:
            st.markdown("<h1>📖</h1>", unsafe_allow_html=True)
            
    with col_info:
        st.markdown(f"### {row.get('title', 'N/A')}")
        st.markdown(f"✍️ **{row.get('author', 'N/A')}**")
        genres = str(row.get('genres', ''))
        if genres and genres != 'nan':
            st.markdown(' '.join([f'`{g.strip()}`' for g in genres.split(',')[:4]]))
        c1, c2 = st.columns(2)
        c1.metric("⭐ Avg Rating", f"{row.get('avg_rating', 'N/A')}")
        if score_label and score_value is not None:
            c2.metric(f"🎯 {score_label}", f"{score_value:.2f}")
        desc = str(row.get('description', ''))
        if desc and desc != 'nan':
            with st.expander("📄 Xem mô tả"):
                st.write(desc[:500] + "..." if len(desc) > 500 else desc)
    st.divider()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📚 Book Recommendation System")
st.markdown("*Hybrid Recommender: Content-Based + Collaborative Filtering (SVDpp)*")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "👤 Gợi ý theo User ID",
    "🔍 Tìm sách tương tự",
    "⭐ Tôi là người dùng mới"   # ← Tab mới
])


# ── Tab 1: User ID ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Gợi ý cá nhân hóa theo User ID")
    st.caption("Dành cho người dùng đã có lịch sử đọc sách trong hệ thống.")
    user_input = st.text_input("Nhập User ID", placeholder="VD: 3672777")

    if st.button("🚀 Gợi ý cho tôi"):
        if not user_input.strip():
            st.warning("Vui lòng nhập User ID!")
        else:
            try:
                # ── Kiểm tra user có trong trainset không ────────────────────
                try:
                    svdpp_model.trainset.to_inner_uid(int(user_input))
                    user_exists = True
                except ValueError:
                    user_exists = False

                # ── Nếu không có → thông báo và dừng ────────────────────────
                if not user_exists:
                    st.warning(
                        f"⚠️ User ID **{user_input}** chưa có trong hệ thống.\n\n"
                        f"👉 Vui lòng sang tab **⭐ Tôi là người dùng mới** "
                        f"để nhận gợi ý dựa trên sở thích của bạn!"
                    )
                else:
                    with st.spinner("Đang tính toán..."):
                        df_rec, is_cold, n_rated = recommend_hybrid(
                            int(user_input),
                            top_n=top_n,
                            w_cf=w_cf,
                            w_cb=w_cb
                        )
                    st.success(f"👤 {n_rated} ratings → Weighted Hybrid (CF={w_cf}, CB={w_cb})")
                    for _, row in df_rec.iterrows():
                        render_book_card(row, "Hybrid Score", row.get('score_hybrid'))

            except Exception as e:
                import traceback
                st.error(f"Lỗi: {e}")
                st.code(traceback.format_exc())

# ── Tab 2: Tìm sách tương tự ──────────────────────────────────────────────────
with tab2:
    st.subheader("Tìm sách tương tự theo tên")
    book_select = st.selectbox(
        "Chọn hoặc gõ tên sách",
        options=[""] + sorted(df_books['title'].dropna().unique().tolist())
    )
    if st.button("🔍 Tìm sách tương tự"):
        if not book_select:
            st.warning("Vui lòng chọn một cuốn sách!")
        else:
            with st.spinner("Đang tìm..."):
                df_sim = recommend_by_title(book_select, top_n=top_n)
            if df_sim is None:
                st.error("Không tìm thấy sách!")
            else:
                st.markdown("#### 📖 Sách bạn chọn")
                render_book_card(df_books[df_books['title'] == book_select].iloc[0])
                st.markdown(f"#### 🎯 {top_n} sách tương tự")
                for _, row in df_sim.iterrows():
                    render_book_card(row, "Similarity", row.get('similarity'))
# ── Tab 3: User mới chọn sách yêu thích ──────────────────────────────────────
with tab3:
    st.subheader("Chọn sách bạn đã đọc và thích")
    st.markdown("Chọn ít nhất **5 cuốn** để nhận gợi ý cá nhân hóa.")

    all_titles = sorted(df_books['title'].dropna().unique().tolist())

    # Multiselect cho user chọn sách
    selected_titles = st.multiselect(
        "🔎 Tìm và chọn sách bạn đã đọc:",
        options=all_titles,
        placeholder="Gõ tên sách để tìm kiếm..."
    )

    # Hiển thị slider rating cho từng sách đã chọn
    user_ratings = {}
    if selected_titles:
        st.markdown("#### ⭐ Bạn đánh giá những sách này bao nhiêu sao?")
        cols = st.columns(min(len(selected_titles), 5))
        for i, title in enumerate(selected_titles):
            with cols[i % 3]:
                rating = st.slider(
                    label=title[:40] + "..." if len(title) > 40 else title,
                    min_value=1,
                    max_value=5,
                    value=4,
                    key=f"rating_{title}"
                )
                user_ratings[title] = rating

    # Nút recommend
    if st.button("🚀 Gợi ý dựa trên sở thích của tôi", key="btn_new_user"):
        if len(selected_titles) < 5:
            st.warning("Vui lòng chọn ít nhất 5 cuốn sách!")
        else:
            # Lấy book_id từ title đã chọn
            title_to_id = pd.Series(
                df_books['book_id'].values,
                index=df_books['title']
            )
            rated_book_ids = []
            for title in selected_titles:
                if title in title_to_id.index:
                    rated_book_ids.append(str(title_to_id[title]))

            with st.spinner("Đang phân tích sở thích của bạn..."):
                # ── Dùng CB từ sách đã chọn (vì user mới không có trong trainset) ──
                cb_scores = {}
                for title, bid in zip(selected_titles, rated_book_ids):
                    weight = user_ratings.get(title, 3) / 5.0  # dùng rating làm trọng số
                    similar = get_content_scores(bid, top_n=100)
                    for sim_bid, score in similar.items():
                        cb_scores[sim_bid] = max(cb_scores.get(sim_bid, 0), score * weight)

                # Normalize CB → [1, 5]
                if cb_scores:
                    min_cb   = min(cb_scores.values())
                    max_cb   = max(cb_scores.values())
                    cb_range = max_cb - min_cb if max_cb != min_cb else 1
                    cb_scores = {
                        k: 1 + 4 * (v - min_cb) / cb_range
                        for k, v in cb_scores.items()
                    }

                # Loại sách đã chọn khỏi kết quả
                candidate_ids = set(df_books['book_id']) - set(rated_book_ids)

                results = []
                for book_id in candidate_ids:
                    bid  = str(book_id)
                    s_cb = cb_scores.get(bid, 1.0)
                    results.append({
                        'book_id'      : bid,
                        'score_hybrid' : round(s_cb, 4),
                    })

                df_new_user = (
                    pd.DataFrame(results)
                    .sort_values('score_hybrid', ascending=False)
                    .head(top_n)
                    .merge(
                        df_books[['book_id', 'title', 'author',
                                  'genres', 'avg_rating', 'image_url']],
                        on='book_id', how='left'
                    )
                    .reset_index(drop=True)
                )

            st.success(f"✅ Dựa trên {len(selected_titles)} sách bạn chọn → Top {top_n} gợi ý:")
                  
            # Hiển thị sách đã chọn
            with st.expander("📚 Sách bạn đã chọn"):
                for title in selected_titles:
                    st.markdown(f"- ⭐ {user_ratings[title]}/5 — **{title}**")

            st.divider()
            for _, row in df_new_user.iterrows():
                render_book_card(row, "CB Score", row.get('score_hybrid'))
               # Lưu rating mới của user vào file để retrain sau
            new_rows = pd.DataFrame([
                {
                    'user_id' : f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                    'book_id' : int(title_to_id[title]),
                    'rating'  : user_ratings[title]
                }
                for title in selected_titles
                if title in title_to_id.index
            ])
            
            new_rows.to_csv(
                'new_ratings.csv',
                mode='a',                          # append — không ghi đè
                header=not pd.io.common.file_exists('new_ratings.csv'),
                index=False
            )
            if save_rating_to_sheets(new_rows):
                st.caption("✅ Sở thích của bạn đã được lưu lên Cloud để học hỏi!")
            else:
                st.caption("⚠️ Không thể lưu lên Cloud, nhưng hệ thống vẫn ghi nhận cục bộ.")
                    