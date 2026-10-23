
"""
Autenticazione Google

Locale: LOCAL_DEV=true -> bypass autenticazione.
Produzione: login Google e controllo email autorizzate.
"""

from __future__ import annotations

import os
import streamlit as st


LOCAL_DEV = os.getenv("LOCAL_DEV", "false").strip().lower() == "true"


def secret_section(name: str) -> dict:
    try:
        return dict(st.secrets.get(name, {}))
    except Exception:
        return {}


def auth_enabled() -> bool:
    return not LOCAL_DEV


def require_login() -> str | None:

    # Ambiente locale di sviluppo
    if LOCAL_DEV:
        return "f.salatesantone@gmail.com"

    # Email autorizzate configurate in Streamlit Secrets
    authorized_emails = {
        str(email).strip().lower()
        for email in secret_section("authorized_emails").get("emails", [])
    }

    if not authorized_emails and not authorized_domains:
        st.error("Nessuna email o dominio autorizzato configurato.")
        st.stop()

    # Verifica configurazione OAuth
    if not secret_section("auth"):
        st.error("Autenticazione Google non configurata nei Secrets.")
        st.stop()

    # Login Google
    if not st.user.is_logged_in:
        st.markdown("""
            <style>
            :root {
                --primary-color: #1769AA;
            }

            div.stButton > button[kind="primary"] {
                background-color: #1769AA !important;
                border-color: #1769AA !important;
                color: white !important;
            }

            div.stButton > button[kind="primary"]:hover {
                background-color: #10558B !important;
                border-color: #10558B !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # Pagina login centrata
        col_left, col_center, col_right = st.columns([1, 1.4, 1])

        with col_center:
            from pathlib import Path
            logo_path = (Path(__file__).resolve().parent.parent / "img" / "logo.png")

            if logo_path.exists():
                st.image(str(logo_path), width="stretch")
            else:
                st.title("AnonymAIzer")

            st.markdown(
                "<h3 style='text-align:center;'>"
                "Centro Studi Tagliacarne"
                "</h3>", unsafe_allow_html=True)

            st.markdown(
                "<p style='text-align:center; color:#64748b;'>"
                "Accedi con il tuo account Google autorizzato."
                "</p>", unsafe_allow_html=True)

            # Pulsante compatto e centrato
            b1, b2, b3 = st.columns([1, 2, 1])
            with b2:
                if st.button("Accedi con Google", type="primary", use_container_width=True):
                    st.login()
        st.stop()

    # Verifica email
    user_email = str(st.user.get("email", "") or "").strip().lower()

    # Domini autorizzati
    authorized_domains = {
        str(domain).strip().lower().lstrip("@")
        for domain in secret_section("authorized_emails").get("domains", [])
    }

    # Verifica email e dominio
    email_domain = user_email.rsplit("@", 1)[-1] if "@" in user_email else ""
    is_authorized = (
        user_email in authorized_emails
        or email_domain in authorized_domains
    )

    if not is_authorized:
        st.error(f"L'account {user_email} non è autorizzato.")

        if st.button("Esci"):
            st.logout()

        st.stop()

    return user_email
